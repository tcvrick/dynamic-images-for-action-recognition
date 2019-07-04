class AverageMeter:

    def __init__(self):
        self.total_measured = 0
        self.total_num_samples = 0

    def add(self, value, num_samples):
        self.total_measured += (value * num_samples)
        self.total_num_samples += num_samples

    def get_average(self):
        return self.total_measured / self.total_num_samples

    def reset(self):
        self.__init__()
        return self


def main():
    pass


if __name__ == '__main__':
    main()
