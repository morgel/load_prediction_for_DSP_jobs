class Printer():
    def __init(self):
        self.unique_log_name = ""  # to override default
        self.verbose = False

    def _print_line(self, *args):
        my_line = ' '.join(str(i) for i in args)
        with open(f"{self.unique_log_name}_log.txt", mode='a') as file_object:
            print(my_line)
            print(my_line, file=file_object)

    def _print_csv(self, *args):
        my_line = '|'.join(str(i) for i in args)
        with open(f"{self.unique_log_name}_results.csv", mode='a') as file_object:
            if self.verbose:
                print(my_line)
            print(my_line, file=file_object)