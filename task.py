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
        "15-1":
            {
                0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                1: [16],
                2: [17],
                3: [18],
                4: [19],
                5: [20]
            },
        "5-0":
            {
                0: [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                1: [1, 2, 3, 4, 5]
            },
        "5-1":
            {
                0: [1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                1: [6, 7, 8, 9, 10]
            },
        "5-2":
            {
                0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17, 18, 19, 20],
                1: [11, 12, 13, 14, 15]
            },
        "5-3":
            {
                0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                1: [16, 17, 18, 19, 20]
            },
        "10-5":
            {
                0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                1: [11, 12, 13, 14, 15],
                2: [16, 17, 18, 19, 20]
            },
        "12-3":
            {
                0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                1: [13, 14, 15]
            }
    },
    "cts": {
        "offline":
            {
                0: [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33],
            },
        "bv":
            {
                0: [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 32, 33],
                1: [27, 28, 29, 30, 31]
            },
    },
    "coco": {
        "offline":
            {
                0: list(range(1, 92)),
            },
        "voc":
            {
                0: [1, 8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                    43, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78,
                    79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90],  # 53589 train img
                1: [2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]  # voc classes w/out person
            },
        "spn":
            {
                0: [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                    15, 16, 17, 18, 19, 20, 23, 24, 26, 27, 28, 29, 30,
                    31, 32, 35, 36, 37, 38, 40, 42, 43, 44, 45, 47, 48,
                    49, 50, 51, 53, 54, 55, 56, 58, 59, 60, 61, 63, 64,
                    65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 77, 78,
                    79, 80, 81, 83, 84, 85, 86, 88, 89, 90],
                1: [21, 25, 33, 34, 41, 57, 87]
            },
        "20":
            {
                0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31,
                    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                    58, 59, 60, 61, 62, 63, 64, 65],  # 79657 train img
                1: [67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
            }},
    "coco-stuff": {
        "offline":
            {
                0: list(range(0, 150)),
            },
        "spn":
            {
                0: [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                    14, 15, 16, 17, 18, 19, 22, 23, 25, 26, 27, 28, 29,
                    30, 31, 34, 35, 36, 37, 39, 41, 42, 43, 44, 46, 47,
                    48, 49, 50, 52, 53, 54, 55, 57, 58, 59, 60, 62, 63,
                    64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77,
                    78, 79, 80, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92,
                    94, 95, 96, 97, 98, 100, 101, 102, 103, 104, 106, 107, 108,
                    109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
                    122, 124, 125, 126, 127, 128, 130, 131, 132, 133, 134, 135, 137,
                    138, 140, 141, 142, 143, 145, 146, 149, 150, 151, 152, 153, 154,
                    155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,
                    169, 170, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181],
                1: [20, 24, 32, 33, 40, 56, 86, 99, 105, 123, 144, 147, 148, 168, 171]
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

        self.disjoint = True

        self.nshot = opts.nshot if self.step > 0 else -1
        self.ishot = opts.ishot

        self.input_mix = opts.input_mix  # novel / both / seen

        self.num_classes = len(self.order)
        self.use_bkg = False if opts.dataset == 'cts' or 'stuff' in opts.dataset else True
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

    def get_old_labels(self, bkg=True):
        if bkg and self.use_bkg:
            return [self.background_label] + [cl for s in range(self.step) for cl in self.task_dict[s]]
        else:
            return [cl for s in range(self.step) for cl in self.task_dict[s]]

    def get_task_dict(self):
        return {s: self.task_dict[s] for s in range(self.step + 1)}

    def get_n_classes(self):
        r = [len(self.task_dict[s]) for s in range(self.step + 1)]
        if self.use_bkg:
            r[0] += 1
        return r
