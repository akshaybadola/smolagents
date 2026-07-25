from typing import Optional
import ast
import builtins
from itertools import zip_longest

from .utils import BASE_BUILTIN_MODULES, get_source, is_valid_name


_BUILTIN_NAMES = set(vars(builtins))


class MethodChecker(ast.NodeVisitor):
    """
    Checks that a method:
    - only uses defined names
    - contains no local imports
    """

    def __init__(
        self,
        class_attributes: set[str],
        check_imports: bool = True,
        existing_imports: Optional[dict[str, str]] = None,
        existing_from_imports: Optional[dict[str, tuple[str, str]]] = None,
        existing_functions: Optional[set[str]] = None,
    ):
        self.undefined_names = set()
        self.imports = {}
        self.from_imports = {}
        self.assigned_names = set()
        self.arg_names = set()
        self.class_attributes = class_attributes
        self.errors = []
        self.check_imports = check_imports
        self.typing_names = {"Any", "Optional", "Union", "List", "Dict", "Set", "Tuple", "Path"}
        self.defined_classes = set()
        self.defined_functions = set(existing_functions) if existing_functions else set()

        if existing_imports:
            self.imports.update(existing_imports)
        if existing_from_imports:
            self.from_imports.update(existing_from_imports)

    def visit_FunctionDef(self, node):
        """Track function definitions created inside or alongside the scope."""
        self.defined_functions.add(node.name)
        # Store outer state to avoid leaking local args across separate functions
        old_args = self.arg_names
        old_assigned = set(self.assigned_names)

        # Collect function args
        self.visit(node.args)

        self.generic_visit(node)

        self.arg_names = old_args
        self.assigned_names = old_assigned

    def visit_arguments(self, node):
        """Collect function arguments."""
        self.arg_names.update({arg.arg for arg in node.args})
        if node.kwarg:
            self.arg_names.add(node.kwarg.arg)
        if node.vararg:
            self.arg_names.add(node.vararg.arg)
        for arg in node.kwonlyargs:
            self.arg_names.add(arg.arg)

    def visit_Import(self, node):
        for name in node.names:
            actual_name = name.asname or name.name
            self.imports[actual_name] = name.name

    def visit_ImportFrom(self, node):
        module = node.module or ""
        for name in node.names:
            actual_name = name.asname or name.name
            self.from_imports[actual_name] = (module, name.name)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.assigned_names.add(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        self.assigned_names.add(elt.id)

    def visit_AugAssign(self, node):
        self._extract_assigned_names(node.target)
        self.visit(node.value)

    def _extract_assigned_names(self, target):
        if isinstance(target, ast.Name):
            self.assigned_names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._extract_assigned_names(elt)

    def visit_With(self, node):
        """Track aliases in 'with' statements."""
        for item in node.items:
            if item.optional_vars:
                self._extract_assigned_names(item.optional_vars)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        """Track exception aliases."""
        if node.name:
            self.assigned_names.add(node.name)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        """Track annotated assignments."""
        if isinstance(node.target, ast.Name):
            self.assigned_names.add(node.target.id)
        if node.value:
            self.visit(node.value)

    def visit_For(self, node):
        self._extract_assigned_names(node.target)
        self.generic_visit(node)

    def _handle_comprehension_generators(self, generators):
        """Helper method to handle generators in all types of comprehensions."""
        for generator in generators:
            self._extract_assigned_names(generator.target)
            self.visit(generator.iter)
            for if_clause in generator.ifs:
                self.visit(if_clause)

    def visit_ListComp(self, node):
        self._handle_comprehension_generators(node.generators)
        self.visit(node.elt)

    def visit_DictComp(self, node):
        self._handle_comprehension_generators(node.generators)
        self.visit(node.key)
        self.visit(node.value)

    def visit_SetComp(self, node):
        self._handle_comprehension_generators(node.generators)
        self.visit(node.elt)

    def visit_Attribute(self, node):
        """Handle attribute access like self.attr or self._walk(...) safely."""
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            # self.<attr> is valid, check the expression value (self)
            return
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        """Track class definitions."""
        self.defined_classes.add(node.name)
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            if not (
                node.id in _BUILTIN_NAMES
                or node.id in BASE_BUILTIN_MODULES
                or node.id in self.arg_names
                or node.id == "self"
                or node.id in self.class_attributes
                or node.id in self.imports
                or node.id in self.from_imports
                or node.id in self.assigned_names
                or node.id in self.typing_names
                or node.id in self.defined_classes
                or node.id in self.defined_functions
            ):
                self.errors.append(f"Name '{node.id}' is undefined.")

    def visit_Call(self, node):
        # Validate simple direct function calls
        if isinstance(node.func, ast.Name):
            if not (
                node.func.id in _BUILTIN_NAMES
                or node.func.id in BASE_BUILTIN_MODULES
                or node.func.id in self.arg_names
                or node.func.id == "self"
                or node.func.id in self.class_attributes
                or node.func.id in self.imports
                or node.func.id in self.from_imports
                or node.func.id in self.assigned_names
                or node.func.id in self.defined_classes
                or node.func.id in self.defined_functions
            ):
                self.errors.append(f"Name '{node.func.id}' is undefined.")
        self.generic_visit(node)


def validate_tool_attributes(
    cls,
    check_imports: bool = True,
    existing_imports=None,
    existing_functions: Optional[set[str] | list[str]] = None,
) -> None:
    """
    Validates that a Tool class follows the proper patterns.
    """
    if existing_imports is None:
        existing_imports = []

    if existing_functions is None:
        existing_functions = []

    allowed_functions = set(existing_functions) if existing_functions else set()

    class ClassLevelChecker(ast.NodeVisitor):
        def __init__(self):
            self.imported_names = set()
            self.complex_attributes = set()
            self.class_attributes = set()
            self.non_defaults = set()
            self.non_literal_defaults = set()
            self.in_method = False
            self.invalid_attributes = []
            self.method_names = set()

        def visit_FunctionDef(self, node):
            self.method_names.add(node.name)
            if node.name == "__init__":
                self._check_init_function_parameters(node)
            old_context = self.in_method
            self.in_method = True
            self.generic_visit(node)
            self.in_method = old_context

        def visit_Assign(self, node):
            if self.in_method:
                return
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.class_attributes.add(target.id)

            if not all(
                isinstance(val, (ast.Str, ast.Num, ast.Constant, ast.Dict, ast.List, ast.Set))
                for val in ast.walk(node.value)
            ):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.complex_attributes.add(target.id)

            if getattr(node.targets[0], "id", "") == "name":
                if not isinstance(node.value, ast.Constant):
                    self.invalid_attributes.append(f"Class attribute 'name' must be a constant, found '{node.value}'")
                elif not isinstance(node.value.value, str):
                    self.invalid_attributes.append(
                        f"Class attribute 'name' must be a string, found '{node.value.value}'"
                    )
                elif not is_valid_name(node.value.value):
                    self.invalid_attributes.append(
                        f"Class attribute 'name' must be a valid Python identifier and not a reserved keyword, found '{node.value.value}'"
                    )

        def _check_init_function_parameters(self, node):
            for arg, default in reversed(list(zip_longest(reversed(node.args.args), reversed(node.args.defaults)))):
                if default is None:
                    if arg.arg != "self":
                        self.non_defaults.add(arg.arg)
                elif not isinstance(default, (ast.Str, ast.Num, ast.Constant, ast.Dict, ast.List, ast.Set)):
                    self.non_literal_defaults.add(arg.arg)

    class_level_checker = ClassLevelChecker()
    source = get_source(cls)
    tree = ast.parse(source)
    class_node = tree.body[0]
    if not isinstance(class_node, ast.ClassDef):
        raise ValueError("Source code must define a class")
    class_level_checker.visit(class_node)

    errors = []
    if class_level_checker.invalid_attributes:
        errors += class_level_checker.invalid_attributes
    if class_level_checker.complex_attributes:
        errors.append(
            f"Complex attributes should be defined in __init__, not as class attributes: "
            f"{', '.join(class_level_checker.complex_attributes)}"
        )
    if class_level_checker.non_defaults:
        errors.append(
            f"Parameters in __init__ must have default values, found required parameters: "
            f"{', '.join(class_level_checker.non_defaults)}"
        )
    if class_level_checker.non_literal_defaults:
        errors.append(
            f"Parameters in __init__ must have literal default values, found non-literal defaults: "
            f"{', '.join(class_level_checker.non_literal_defaults)}"
        )

    parsed_imports = {}
    parsed_from_imports = {}
    for e in existing_imports:
        t_ = ast.parse(e)
        node = t_.body[0]
        method_checker = MethodChecker(
            class_level_checker.class_attributes,
            existing_imports=parsed_imports,
            existing_from_imports=parsed_from_imports,
            existing_functions=allowed_functions,
        )
        method_checker.visit(node)
        parsed_imports.update(method_checker.imports)
        parsed_from_imports.update(method_checker.from_imports)
        errors += [f"- {error}" for error in method_checker.errors]
        if errors:
            import ipdb; ipdb.set_trace()

    parsed_functions = allowed_functions.union(class_level_checker.method_names)
    for e in existing_functions:
        t_ = ast.parse(e)
        node = t_.body[0]
        method_checker = MethodChecker(
            class_level_checker.class_attributes,
            existing_imports=parsed_imports,
            existing_from_imports=parsed_from_imports,
            existing_functions=parsed_functions,
        )
        method_checker.visit(node)
        parsed_functions.update(method_checker.defined_functions)
        errors += [f"- {error}" for error in method_checker.errors]
        if errors:
            import ipdb; ipdb.set_trace()

    # Run checks on all methods
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef):
            method_checker = MethodChecker(
                class_level_checker.class_attributes,
                check_imports=check_imports,
                existing_imports=parsed_imports,
                existing_from_imports=parsed_from_imports,
                existing_functions=parsed_functions,
            )
            method_checker.visit(node)
            errors += [f"- {node.name}: {error}" for error in method_checker.errors]
            if errors:
                import ipdb; ipdb.set_trace()

    if errors:
        raise ValueError(f"Tool validation failed for {cls.__name__}:\n" + "\n".join(errors))
