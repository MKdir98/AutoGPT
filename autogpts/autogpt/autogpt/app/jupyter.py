from notebook.notebookapp import NotebookApp


def run():
    app = NotebookApp.instance()
    app.port = 8888
    app.notebook_dir = 'notebooks'
    app.initialize(argv=[])
    app.start()

