class processing_step:
    def __init__(self, function, args, kwdargs):
        self.func = function
        self.args = args
        self.kwdargs = kwdargs

    def __call__(self, *data):
        return self.function(*data, *self.args, **self.kwdargs)


class pipeline:
    def __init__(self):
        self.processing_steps = []

    def then(self, function, *params, **kwdparams):
        self.processing_steps.append(processing_step(function, params, kwdparams))

    def __call__(self, *data):
        for step in self.processing_setps:
            data = step(*data)
        return data
        
