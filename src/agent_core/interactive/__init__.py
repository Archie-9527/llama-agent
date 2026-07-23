"""Runtime services shared by interactive front-ends.

Submodules are intentionally not imported here: the model and graph publish
events during their own imports, and eager re-exports would create a circular
dependency through ``ConversationManager``.
"""
