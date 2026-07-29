from time import sleep


class pbar:
    def __init__(self, total, msg, blocks=20, percentage=True):
        self.total = total
        self.msg = msg
        self.percentage = percentage
        self.i = 0
        self.blocks = blocks
        self.delete = ""
        self.next()

    def next(self):
        if self.i > self.total:
            return
        p = round(self.blocks * self.i / self.total)
        bar = "|" + "#" * p + "-" * (self.blocks - p) + "|"
        prog_num = str(self.i) + "/" + str(self.total)
        if self.percentage:
            prog_num = str(round(100 * self.i / self.total, 2)) + "%"
        line = self.msg + bar + prog_num
        print(self.delete + line, end='', flush=True)
        if self.i == self.total:
            print()
        self.delete = "\b" * 2 * len(line)
        self.i += 1


def test_print():
    bar = pbar(10, "testing ")
    for i in range(10):
        bar.next()
        sleep(0.5)
