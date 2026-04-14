"""
Professional Textual-based TUI for Ornamentalist.

Provides a visual, interactive interface to configure hyperparameters,
preview generated JSON config arrays, and run sweeps.

To use, ensure Textual is installed: `pip install textual`
"""

import inspect
import itertools
import json
import types
import typing
from typing import Any, Dict, List

from rich.syntax import Syntax
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.widgets import Button, Collapsible, Footer, Header, Input, Label, Static

import ornamentalist


def parse_value(val_str: str, anno: Any) -> Any:
    """Replicates the type-casting logic from ornamentalist.cli() with strict validation."""
    if anno is inspect.Parameter.empty:
        return val_str  # Fallback to raw string

    origin = typing.get_origin(anno)
    args = typing.get_args(anno)

    # Handle `Type | None` or `Union[Type, None]`
    if origin is types.UnionType or origin is typing.Union:
        if val_str.lower() in ("none", "null"):
            return None
        arg_type = [t for t in args if t is not type(None)][0]
        try:
            return arg_type(val_str)
        except ValueError:
            raise ValueError(f"Cannot cast '{val_str}' to {arg_type.__name__}")

    # Handle Literals
    if origin is typing.Literal:
        arg_type = type(args[0])
        try:
            parsed = arg_type(val_str)
            if parsed not in args:
                raise ValueError(f"'{parsed}' is not a valid choice. Allowed: {args}")
            return parsed
        except ValueError:
            raise ValueError(
                f"Cannot cast '{val_str}' to {arg_type.__name__} for Literal"
            )

    # Handle Booleans
    if anno is bool:
        if val_str.lower() in ("true", "t", "1"):
            return True
        if val_str.lower() in ("false", "f", "0"):
            return False
        raise ValueError(f"Invalid boolean value: '{val_str}'. Use True/False.")

    # Handle built-ins
    if anno in (int, float, str):
        try:
            return anno(val_str)
        except ValueError:
            raise ValueError(f"Cannot cast '{val_str}' to {anno.__name__}")

    # Fallback
    return val_str


class ParamInput(Horizontal):
    """A layout row for a single parameter input."""

    def __init__(
        self, fn_name: str, param_name: str, param: inspect.Parameter, default_val: str
    ):
        super().__init__(classes="param-row")
        self.fn_name = fn_name
        self.param_name = param_name
        self.anno = param.annotation

        # Format the type hint for the UI label
        type_str = getattr(self.anno, "__name__", str(self.anno)).replace("typing.", "")
        if typing.get_origin(self.anno) is typing.Literal:
            type_str = f"Literal{typing.get_args(self.anno)}"
        elif self.anno is inspect.Parameter.empty:
            type_str = "str (fallback)"
        elif typing.get_origin(self.anno) in (types.UnionType, typing.Union):
            type_str = " | ".join(
                [getattr(t, "__name__", str(t)) for t in typing.get_args(self.anno)]
            )

        self.label_str = f"{param_name}\n[dim]{type_str}[/dim]"
        self.default_val = default_val

    def compose(self) -> ComposeResult:
        yield Label(self.label_str, classes="param-label")
        inp = Input(
            value=self.default_val,
            placeholder="e.g. 1, 2, 3 (comma-separated for sweeps)",
            classes="param-input",
        )
        # Store metadata on the input widget for easy retrieval during validation
        inp.fn_name = self.fn_name
        inp.param_name = self.param_name
        inp.anno = self.anno
        yield inp


class OrnamentalApp(App):
    """The main Textual application."""

    TITLE = "Ornamentalist | Configuration & Sweep Manager"
    CSS = """
    Screen {
        background: $surface-darken-1;
    }
    #main-container {
        height: 1fr;
    }
    #left-pane {
        width: 55%;
        height: 100%;
        padding: 1 2;
        border-right: vkey $background;
    }
    #right-pane {
        width: 45%;
        height: 100%;
        padding: 1 2;
        background: $surface;
    }
    .param-row {
        height: auto;
        margin-bottom: 1;
        align: left middle;
    }
    .param-label {
        width: 35%;
        text-align: right;
        padding-right: 2;
        color: $text-muted;
    }
    .param-input {
        width: 65%;
    }
    #run-stats {
        height: auto;
        padding: 1 2;
        border: solid $success;
        background: $panel;
        margin-bottom: 1;
    }
    #run-stats.error {
        border: solid $error;
        color: $error;
    }
    #json-preview {
        height: 1fr;
        padding: 1 2;
        border: solid $primary;
        background: $panel;
    }
    #button-container {
        align: center middle;
        height: auto;
        margin-top: 2;
        margin-bottom: 1;
    }
    Button {
        width: 100%;
    }
    """

    BINDINGS = [
        ("ctrl+r", "run_configs", "Run Selected Configs"),
        ("ctrl+q", "quit", "Quit (Cancel)"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()

        with Horizontal(id="main-container"):
            # Left Pane: Inputs
            with ScrollableContainer(id="left-pane"):
                if not ornamentalist._CONFIGURABLE_FUNCTIONS:
                    yield Label(
                        "[yellow]No @ornamentalist.configure() functions found.[/yellow]"
                    )

                for f in ornamentalist._CONFIGURABLE_FUNCTIONS:
                    with Collapsible(title=f"⚙️  [b]{f.name}[/b]", collapsed=False):
                        for param_name in f.params_to_inject:
                            param = f.signature.parameters[param_name]

                            default_val = ""
                            if f.cli_defaults and param_name in f.cli_defaults:
                                default_val = str(f.cli_defaults[param_name])

                            yield ParamInput(f.name, param_name, param, default_val)

                with Horizontal(id="button-container"):
                    yield Button(
                        "🚀 Generate & Run (Ctrl+R)", id="run_btn", variant="success"
                    )

            # Right Pane: Live Visualization & Stats
            with Vertical(id="right-pane"):
                yield Static("[b]Total Runs Queued:[/b] 0", id="run-stats")
                yield ScrollableContainer(Static(id="json-preview"))

        yield Footer()

    def on_mount(self) -> None:
        """Trigger an initial render of the preview pane when the app starts."""
        self.update_preview()

    @on(Input.Changed)
    def on_input_changed(self, event: Input.Changed) -> None:
        """React to any keystroke in any input field to update the live preview."""
        self.update_preview()

    def action_run_configs(self):
        self.execute_run()

    @on(Button.Pressed, "#run_btn")
    def on_run_button_pressed(self, event: Button.Pressed):
        self.execute_run()

    def _build_configs(self) -> List[Dict[str, Any]]:
        """Parses the UI inputs and computes the Cartesian product."""
        parsed_data = {}

        for inp in self.query(Input):
            fn_name = getattr(inp, "fn_name", None)
            if not fn_name:
                continue

            param_name = inp.param_name
            anno = inp.anno

            if fn_name not in parsed_data:
                parsed_data[fn_name] = {}

            val_str = inp.value.strip()
            vals = [v.strip() for v in val_str.split(",") if v.strip()]

            if not vals:
                raise ValueError(f"Missing required value for {fn_name}.{param_name}")

            parsed_vals = [parse_value(v, anno) for v in vals]
            parsed_data[fn_name][param_name] = parsed_vals

        # Compute the Cartesian product across all parameters
        flat_params = []
        flat_lists = []
        for fn_name, params in parsed_data.items():
            for param_name, vals_list in params.items():
                flat_params.append((fn_name, param_name))
                flat_lists.append(vals_list)

        configs = []
        for combo in itertools.product(*flat_lists):
            config_dict = {}
            for (fn_name, param_name), val in zip(flat_params, combo):
                if fn_name not in config_dict:
                    config_dict[fn_name] = {}
                config_dict[fn_name][param_name] = val
            configs.append(config_dict)

        return configs

    def update_preview(self) -> None:
        """Updates the right pane with the current run count and JSON preview."""
        stats_box = self.query_one("#run-stats", Static)
        preview_box = self.query_one("#json-preview", Static)
        run_btn = self.query_one("#run_btn", Button)

        try:
            configs = self._build_configs()
            num_runs = len(configs)

            stats_box.update(f"🚀 [b]Total Runs Queued:[/b] {num_runs}")
            stats_box.remove_class("error")
            run_btn.disabled = False

            # Format JSON for preview
            if num_runs == 0:
                json_str = "[]"
            elif num_runs > 10:
                # Truncate preview for performance and readability if massive sweeps are queued
                preview_configs = configs[:10]
                json_str = json.dumps(preview_configs, indent=2)
                json_str = (
                    json_str[:-1]
                    + f"  ...\n  // (and {num_runs - 10} more configurations hidden)\n]"
                )
            else:
                json_str = json.dumps(configs, indent=2)

            # Use Rich's Syntax highlighter for a professional aesthetic
            syntax = Syntax(
                json_str,
                "json",
                theme="monokai",
                background_color="default",
                word_wrap=True,
            )
            preview_box.update(syntax)

        except ValueError as e:
            # Handle validation errors gracefully
            stats_box.update(f"⚠️  [b]Validation Error:[/b] Cannot launch runs.")
            stats_box.add_class("error")
            preview_box.update(
                f"[red]{str(e)}[/red]\n\nPlease fix the input on the left."
            )
            run_btn.disabled = True

    def execute_run(self):
        """Finalize and return configs to the main thread."""
        try:
            configs = self._build_configs()
            self.exit(result=configs)
        except ValueError:
            self.notify("Cannot run with validation errors.", severity="error")


def tui() -> list[ornamentalist.ConfigDict]:
    """
    Launch the Textual UI, parse user inputs, and return the configurations.
    Returns an empty list if the user quits without running.
    """
    app = OrnamentalApp()
    result = app.run()
    return result if result else []


if __name__ == "__main__":
    from typing import Literal

    from ornamentalist import Configurable

    @ornamentalist.configure()
    def multiply(
        a: int = Configurable[5],
        b: Literal[2, 3, 4] = Configurable,
        c: int | None = Configurable[None],
    ):
        if c is None:
            print(a * b)
        else:
            print(a * b * c)

    configs = tui()
    for config in configs:
        ornamentalist.setup(config, force=True)
        multiply()
