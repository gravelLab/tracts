def pytest_configure(config):
    """
    Registers the pyparsing-deprecation-warning filter that would otherwise live in
    pyproject.toml's [tool.pytest.ini_options] filterwarnings list. That list resolves each
    "ignore::some.dotted.Category" entry by importing the module and looking up the class at
    pytest-collection time; since pyparsing's PyparsingDeprecationWarning class has moved between
    a top-level re-export and its pyparsing.warnings submodule across versions, an environment
    with a version lacking whichever path is hardcoded crashes pytest collection entirely
    (AttributeError/ModuleNotFoundError) instead of just skipping that one filter.

    Registering it via config.addinivalue_line (rather than calling warnings.filterwarnings()
    directly) matters: pytest wraps every test in warnings.catch_warnings() and re-derives its
    filter list from config.getini("filterwarnings") each time, inserting pytest's own "always"
    DeprecationWarning default afterwards. PyparsingDeprecationWarning subclasses
    DeprecationWarning, so a filter added once via a bare warnings.filterwarnings() call here
    ends up behind that "always" default and gets shadowed. Feeding it through addinivalue_line
    makes it part of the same ini-sourced list pytest re-applies (in the right position) for
    every test, so it actually takes effect.
    """
    for dotted_path in ("pyparsing.PyparsingDeprecationWarning", "pyparsing.warnings.PyparsingDeprecationWarning"):
        module_name, _, class_name = dotted_path.rpartition(".")
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
        except (ImportError, AttributeError):
            continue
        config.addinivalue_line("filterwarnings", f"ignore::{dotted_path}")
        break
