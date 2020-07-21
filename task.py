tasks = {
    "voc": {
        "offline":
            {
                0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            },
        "15-5":
            {
                0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                1: [16, 17, 18, 19, 20]
            },
        "12-3":
            {
                0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                1: [13, 14, 15]
            }
    }
}


def get_task_list():
    return [task for ds in tasks.keys() for task in tasks[ds].keys()]


class Task:
    def __init__(self, opts):
        self.step = opts.step
        self.dataset = opts.dataset
        self.task = opts.task
        if self.task not in tasks[self.dataset]:
            raise NotImplementedError(f"The task {self.task} is not present in {self.dataset}")
        self.task_dict = tasks[self.dataset][self.task]
        assert self.step in self.task_dict.keys(), f"You should provide a valid step! [{self.step} is out of range]"
        self.order = [cl for s in range(self.step + 1) for cl in self.task_dict[s]]

        self.nshot = opts.nshot if self.step > 0 else -1
        self.ishot = opts.ishot

        self.input_mix = opts.input_mix  # novel / both / seen

        self.num_classes = len(self.order)
        self.use_bkg = opts.use_bkg
        if self.use_bkg:
            self.order.insert(0, 0)
            self.background_label = 0
            self.num_classes += 1
        else:
            self.background_label = 255

    def get_order(self):
        return self.order

    def get_novel_labels(self):
        return list(self.task_dict[self.step])

    def get_old_labels(self):
        return [cl for s in range(self.step) for cl in self.task_dict[s]]

    def get_task_dict(self):
        return {s: self.task_dict[s] for s in range(self.step+1)}

    def get_n_classes(self):
        r = [len(self.task_dict[s]) for s in range(self.step+1)]
        if self.use_bkg:
            r[0] += 1
        return r
