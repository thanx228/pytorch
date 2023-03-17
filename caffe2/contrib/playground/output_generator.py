




from caffe2.python import timeout_guard

def fun_conclude_operator(self):
    # Ensure the program exists. This is to "fix" some unknown problems
    # causing the job sometimes get stuck.
    timeout_guard.EuthanizeIfNecessary(600.0)


def assembleAllOutputs(self):
    return {
        'train_model': self.train_model,
        'test_model': self.test_model,
        'model': self.model_output,
        'metrics': self.metrics_output,
    }
