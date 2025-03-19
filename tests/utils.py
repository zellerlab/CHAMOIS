try:
    from importlib.resources import files as resource_files
except ImportError:
    try:
        from importlib_resources import files as resource_files
    except ImportError:
        resource_files = None