"""
Professional Textual-based TUI for Ornamentalist.

Provides an interactive interface to configure hyperparameters,
preview generated configurations, and launch sweeps.

Requires Textual: pip install textual
"""

import enum
import inspect
import itertools
import json
import types
import typing
from typing import Any

from rich.syntax import Syntax
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, HorizontalGroup, ScrollableContainer, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Collapsible,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Static,
    TabbedContent,
    TabPane,
)

import ornamentalist


# --- Type classification ---


class ParamKind(enum.Enum):
    INT = "int"
    FLOAT = "float"
    STR = "str"
    BOOL = "bool"
    LITERAL = "literal"
    NULLABLE = "nullable"
    UNKNOWN = "unknown"


def classify_param(annotation: Any) -> tuple[ParamKind, type | None, tuple | None]:
    """Classify a parameter's type annotation into a ParamKind.

    Returns (kind, inner_type, literal_choices).
    """
    if annotation is inspect.Parameter.empty:
        return ParamKind.UNKNOWN, str, None

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    if origin is types.UnionType or origin is typing.Union:
        non_none = [t for t in args if t is not type(None)]
        if len(non_none) == 1 and type(None) in args and non_none[0] in (int, float, str):
            return ParamKind.NULLABLE, non_none[0], None
        return ParamKind.UNKNOWN, str, None

    if origin is typing.Literal and args:
        return ParamKind.LITERAL, type(args[0]), args

    if annotation is bool:
        return ParamKind.BOOL, None, None
    if annotation is int:
        return ParamKind.INT, int, None
    if annotation is float:
        return ParamKind.FLOAT, float, None
    if annotation is str:
        return ParamKind.STR, str, None

    return ParamKind.UNKNOWN, str, None


# --- Widgets ---


class PillButton(Button):
    """A toggleable pill button for selecting values."""

    active = reactive(False)

    def __init__(self, label: str, active: bool = False, **kwargs):
        self._ready = False
        super().__init__(label, **kwargs)
        self.active = active

    def on_mount(self) -> None:
        self._ready = True
        self.set_class(self.active, "pill-on")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.active = not self.active

    def watch_active(self, active: bool) -> None:
        self.set_class(active, "pill-on")
        if getattr(self, "_ready", False):
            self.post_message(self.Changed(self))

    class Changed(Message):
        """Posted when the pill is toggled."""

        def __init__(self, pill: "PillButton") -> None:
            super().__init__()
            self.pill = pill


class ParamWidget(Vertical):
    """Composite widget for configuring a single parameter.

    Renders the appropriate input control based on the parameter's type
    and exposes get_values() to extract parsed value(s) for sweep support.
    """

    def __init__(
        self,
        fn_name: str,
        param_name: str,
        kind: ParamKind,
        inner_type: type | None,
        choices: tuple | None,
        default: Any,
        required: bool,
    ):
        safe_id = f"{fn_name}--{param_name}".replace(".", "-")
        super().__init__(classes="param-widget", id=f"pw-{safe_id}")
        self.fn_name = fn_name
        self.param_name = param_name
        self.kind = kind
        self.inner_type = inner_type
        self.choices = choices
        self.default = default
        self.required = required
        self._safe_id = safe_id

    @property
    def _type_label(self) -> str:
        match self.kind:
            case ParamKind.LITERAL:
                return f"Literal{list(self.choices)}"
            case ParamKind.NULLABLE:
                return f"{(self.inner_type or str).__name__} | None"
            case ParamKind.UNKNOWN:
                return "str (inferred)"
            case ParamKind.BOOL:
                return "bool"
            case _:
                return (self.inner_type or str).__name__

    def compose(self) -> ComposeResult:
        badge = "required" if self.required else f"default: {self.default}"
        yield Label(
            f"[b]{self.param_name}[/b]  "
            f"[dim]{self._type_label}[/dim]  "
            f"[dim italic]({badge})[/dim italic]",
            classes="param-header",
        )

        match self.kind:
            case ParamKind.BOOL:
                default_val = self.default if isinstance(self.default, bool) else False
                with HorizontalGroup(classes="toggle-row"):
                    yield PillButton(
                        "True", active=default_val,
                        id=f"pill-{self._safe_id}-true",
                    )
                    yield PillButton(
                        "False", active=not default_val,
                        id=f"pill-{self._safe_id}-false",
                    )

            case ParamKind.LITERAL:
                with HorizontalGroup(classes="toggle-row"):
                    for choice in self.choices:
                        yield PillButton(
                            str(choice), active=(choice == self.default),
                            id=f"pill-{self._safe_id}-{choice}",
                        )

            case ParamKind.NULLABLE:
                is_null = self.default is None
                inner_val = "" if is_null else str(self.default)
                type_name = (self.inner_type or str).__name__
                with HorizontalGroup(classes="toggle-row nullable-row"):
                    yield PillButton(
                        "None", active=is_null,
                        id=f"pill-{self._safe_id}-none",
                    )
                    yield Input(
                        value=inner_val,
                        placeholder=f"{type_name} value(s), comma-sep for sweep",
                        id=f"inp-{self._safe_id}",
                    )

            case _:  # INT, FLOAT, STR, UNKNOWN
                default_str = str(self.default) if self.default is not None else ""
                type_name = (self.inner_type or str).__name__
                yield Input(
                    value=default_str,
                    placeholder=f"{type_name} value(s), comma-separated for sweep",
                    id=f"inp-{self._safe_id}",
                )

        yield Label("", classes="field-error", id=f"err-{self._safe_id}")

    def get_values(self) -> list[Any]:
        """Parse and return configured value(s). Always returns a list for sweep support."""
        match self.kind:
            case ParamKind.BOOL:
                true_pill = self.query_one(f"#pill-{self._safe_id}-true", PillButton)
                false_pill = self.query_one(f"#pill-{self._safe_id}-false", PillButton)
                values = []
                if true_pill.active:
                    values.append(True)
                if false_pill.active:
                    values.append(False)
                if not values:
                    raise ValueError(
                        f"Select at least one value for '{self.param_name}'"
                    )
                return values

            case ParamKind.LITERAL:
                values = []
                for choice in self.choices:
                    pill = self.query_one(
                        f"#pill-{self._safe_id}-{choice}", PillButton
                    )
                    if pill.active:
                        values.append(choice)
                if not values:
                    raise ValueError(
                        f"Select at least one value for '{self.param_name}'"
                    )
                return values

            case ParamKind.NULLABLE:
                none_pill = self.query_one(
                    f"#pill-{self._safe_id}-none", PillButton
                )
                inp = self.query_one(Input)
                values: list[Any] = []
                if none_pill.active:
                    values.append(None)
                text = inp.value.strip()
                if text:
                    cast = self.inner_type or str
                    for token in text.split(","):
                        token = token.strip()
                        if not token:
                            continue
                        try:
                            values.append(cast(token))
                        except (ValueError, TypeError):
                            raise ValueError(
                                f"'{token}' is not a valid {cast.__name__}"
                            )
                if not values:
                    raise ValueError(
                        f"Provide a value or check 'null' for '{self.param_name}'"
                    )
                return values

            case _:  # INT, FLOAT, STR, UNKNOWN
                inp = self.query_one(Input)
                text = inp.value.strip()
                if not text:
                    raise ValueError(f"Value required for '{self.param_name}'")
                cast = self.inner_type or str
                values = []
                for token in text.split(","):
                    token = token.strip()
                    if not token:
                        continue
                    try:
                        values.append(cast(token))
                    except (ValueError, TypeError):
                        raise ValueError(
                            f"'{token}' is not a valid {cast.__name__}"
                        )
                if not values:
                    raise ValueError(f"Value required for '{self.param_name}'")
                return values

    def show_error(self, message: str) -> None:
        err = self.query_one(f"#err-{self._safe_id}", Label)
        err.update(message)
        err.add_class("show")

    def clear_error(self) -> None:
        err = self.query_one(f"#err-{self._safe_id}", Label)
        err.update("")
        err.remove_class("show")


class ConfirmRunScreen(ModalScreen[bool]):
    """Confirmation modal for launching large sweeps."""

    CSS = """
    ConfirmRunScreen {
        align: center middle;
    }
    #confirm-dialog {
        width: 60;
        height: auto;
        padding: 2 4;
        border: thick $primary;
        background: $surface;
    }
    #confirm-title {
        width: 100%;
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    #confirm-body {
        width: 100%;
        text-align: center;
        margin-bottom: 2;
    }
    #confirm-buttons {
        height: auto;
        align: center middle;
    }
    #confirm-buttons Button {
        margin: 0 2;
    }
    """

    def __init__(self, num_configs: int):
        super().__init__()
        self.num_configs = num_configs

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Label("Confirm Launch", id="confirm-title")
            yield Label(
                f"This will generate [b]{self.num_configs}[/b] configurations.\n"
                "Proceed?",
                id="confirm-body",
            )
            with Horizontal(id="confirm-buttons"):
                yield Button("Launch", variant="success", id="confirm-yes")
                yield Button("Cancel", variant="default", id="confirm-no")

    @on(Button.Pressed, "#confirm-yes")
    def on_confirm(self, event: Button.Pressed) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#confirm-no")
    def on_cancel(self, event: Button.Pressed) -> None:
        self.dismiss(False)

    def key_escape(self) -> None:
        self.dismiss(False)


# --- Main Application ---


class OrnamentalApp(App):
    """Ornamentalist configuration TUI."""

    TITLE = "ornamentalist"

    CSS = """
    Screen {
        background: $surface-darken-1;
    }

    /* Main layout */
    #main {
        height: 1fr;
    }
    #left {
        width: 55%;
        height: 100%;
        padding: 1 2;
        border-right: tall $surface-lighten-1;
    }
    #right {
        width: 45%;
        height: 100%;
        padding: 1 2;
    }

    /* Parameter widgets */
    .param-widget {
        height: auto;
        padding: 1;
        margin-bottom: 1;
    }
    .param-widget.sweep {
        border-left: thick $primary;
    }
    .param-header {
        height: auto;
        margin-bottom: 1;
    }
    .field-error {
        color: $error;
        height: auto;
        display: none;
        margin-top: 1;
    }
    .field-error.show {
        display: block;
    }

    /* Toggle pill rows */
    .toggle-row {
        height: auto;
        align: left middle;
    }
    PillButton {
        min-width: 8;
        margin-right: 1;
        border: tall $surface-lighten-2;
        background: $surface;
        color: $text-muted;
    }
    PillButton.pill-on {
        border: tall $primary;
        background: $primary-darken-2;
        color: $text;
        text-style: bold;
    }
    PillButton:focus {
        border: tall $accent;
    }

    /* Nullable row: pill + input side by side */
    .nullable-row Input {
        width: 1fr;
    }

    /* Status bar */
    #status {
        height: auto;
        padding: 1 2;
        margin-bottom: 1;
        border: solid $success;
        background: $panel;
    }
    #status.err {
        border: solid $error;
    }

    /* Button bar */
    #buttons {
        height: auto;
        align: center middle;
        margin-top: 2;
        padding: 1;
    }
    #buttons Button {
        margin: 0 1;
    }

    /* Preview pane */
    DataTable {
        height: 1fr;
    }
    #json-view {
        height: 1fr;
        padding: 1;
    }

    /* Collapsible sections */
    Collapsible {
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        ("ctrl+r", "run_configs", "Run"),
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+j", "toggle_preview", "Toggle Preview"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()

        with Horizontal(id="main"):
            with ScrollableContainer(id="left"):
                if not ornamentalist._CONFIGURABLE_FUNCTIONS:
                    yield Label(
                        "[yellow]No configurable functions found.[/yellow]"
                    )
                else:
                    for f in ornamentalist._CONFIGURABLE_FUNCTIONS:
                        with Collapsible(title=f.name, collapsed=False):
                            for param_name in f.params_to_inject:
                                param = f.signature.parameters[param_name]
                                kind, inner_type, choices = classify_param(
                                    param.annotation
                                )
                                default = None
                                required = True
                                if f.cli_defaults and param_name in f.cli_defaults:
                                    default = f.cli_defaults[param_name]
                                    required = False

                                yield ParamWidget(
                                    fn_name=f.name,
                                    param_name=param_name,
                                    kind=kind,
                                    inner_type=inner_type,
                                    choices=choices,
                                    default=default,
                                    required=required,
                                )

                    with Horizontal(id="buttons"):
                        yield Button(
                            "Generate & Run",
                            variant="success",
                            id="run-btn",
                        )
                        yield Button("Cancel", variant="default", id="cancel-btn")

            with Vertical(id="right"):
                yield Static("[b]0 configurations[/b]", id="status")
                with TabbedContent(id="preview-tabs"):
                    with TabPane("Table", id="table-tab"):
                        yield DataTable(id="config-table", zebra_stripes=True)
                    with TabPane("JSON", id="json-tab"):
                        yield Static(id="json-view")

        yield Footer()

    def on_mount(self) -> None:
        if ornamentalist._CONFIGURABLE_FUNCTIONS:
            table = self.query_one("#config-table", DataTable)
            table.add_column("#")
            for f in ornamentalist._CONFIGURABLE_FUNCTIONS:
                for param_name in f.params_to_inject:
                    table.add_column(f"{f.name}.{param_name}")
            table.cursor_type = "row"
            self.update_preview()

    # --- Event handlers ---

    @on(Input.Changed)
    def on_input_changed(self, event: Input.Changed) -> None:
        self.update_preview()

    @on(PillButton.Changed)
    def on_pill_changed(self, event: PillButton.Changed) -> None:
        self.update_preview()

    def action_run_configs(self) -> None:
        self._execute_run()

    def action_toggle_preview(self) -> None:
        tabs = self.query_one("#preview-tabs", TabbedContent)
        if tabs.active == "table-tab":
            tabs.active = "json-tab"
        else:
            tabs.active = "table-tab"

    @on(Button.Pressed, "#run-btn")
    def on_run_pressed(self, event: Button.Pressed) -> None:
        self._execute_run()

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel_pressed(self, event: Button.Pressed) -> None:
        self.exit(result=None)

    # --- Preview ---

    def update_preview(self) -> None:
        """Rebuild the config preview from current widget state."""
        status = self.query_one("#status", Static)
        run_btn = self.query_one("#run-btn", Button)

        all_params: list[tuple[str, str]] = []
        all_values: list[list[Any]] = []
        has_error = False

        for widget in self.query(ParamWidget):
            try:
                values = widget.get_values()
                widget.clear_error()
                all_params.append((widget.fn_name, widget.param_name))
                all_values.append(values)
                if len(values) > 1:
                    widget.add_class("sweep")
                else:
                    widget.remove_class("sweep")
            except ValueError as e:
                widget.show_error(str(e))
                widget.remove_class("sweep")
                has_error = True

        if has_error:
            status.update("[b]Validation errors[/b] -- fix inputs to continue")
            status.add_class("err")
            run_btn.disabled = True
            self._clear_preview()
            return

        configs = self._build_configs(all_params, all_values)
        n = len(configs)
        label = "configuration" if n == 1 else "configurations"
        status.update(f"[b]{n} {label}[/b] ready to launch")
        status.remove_class("err")
        run_btn.disabled = False
        self._update_table(configs, all_params)
        self._update_json(configs)

    @staticmethod
    def _build_configs(
        all_params: list[tuple[str, str]], all_values: list[list[Any]]
    ) -> list[dict[str, dict[str, Any]]]:
        configs = []
        for combo in itertools.product(*all_values):
            config: dict[str, dict[str, Any]] = {}
            for (fn_name, param_name), value in zip(all_params, combo):
                if fn_name not in config:
                    config[fn_name] = {}
                config[fn_name][param_name] = value
            configs.append(config)
        return configs

    def _clear_preview(self) -> None:
        self.query_one("#config-table", DataTable).clear()
        self.query_one("#json-view", Static).update("")

    def _update_table(
        self,
        configs: list[dict],
        all_params: list[tuple[str, str]],
    ) -> None:
        table = self.query_one("#config-table", DataTable)
        table.clear()
        cap = 50
        for i, config in enumerate(configs[:cap]):
            row = [str(i + 1)]
            for fn_name, param_name in all_params:
                val = config.get(fn_name, {}).get(param_name, "")
                row.append(str(val))
            table.add_row(*row)
        if len(configs) > cap:
            table.add_row(f"... +{len(configs) - cap}", *["" for _ in all_params])

    def _update_json(self, configs: list[dict]) -> None:
        view = self.query_one("#json-view", Static)
        if not configs:
            view.update("[]")
            return
        cap = 50
        json_str = json.dumps(configs[:cap], indent=2)
        if len(configs) > cap:
            json_str = (
                json_str[:-1]
                + f"\n  // ... and {len(configs) - cap} more\n]"
            )
        syntax = Syntax(
            json_str,
            "json",
            theme="monokai",
            background_color="default",
            word_wrap=True,
        )
        view.update(syntax)

    # --- Run ---

    def _execute_run(self) -> None:
        try:
            all_params = []
            all_values = []
            for widget in self.query(ParamWidget):
                values = widget.get_values()
                all_params.append((widget.fn_name, widget.param_name))
                all_values.append(values)
            configs = self._build_configs(all_params, all_values)

            if len(configs) > 5:
                self.push_screen(
                    ConfirmRunScreen(len(configs)),
                    callback=self._on_confirm,
                )
            else:
                self.exit(result=configs)
        except ValueError:
            self.notify("Fix validation errors before running.", severity="error")

    def _on_confirm(self, confirmed: bool) -> None:
        if not confirmed:
            return
        try:
            all_params = []
            all_values = []
            for widget in self.query(ParamWidget):
                values = widget.get_values()
                all_params.append((widget.fn_name, widget.param_name))
                all_values.append(values)
            configs = self._build_configs(all_params, all_values)
            self.exit(result=configs)
        except ValueError:
            self.notify(
                "Configuration changed. Please try again.", severity="error"
            )


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
