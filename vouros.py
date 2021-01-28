delay = [0, 30, 60, 90, 120, 180, 240]
time = [x * 30 for x in range(14)]
a = []


def drive_a_path():
    for i in delay:
        for j in delay:
            if check(i, j) is False:
                break
        break



def check(i, j):
    for t in time:
        if t <= i < t + 60:
            if t <= j < t + 60:
                print((i, j))
            else:
                print((i, j))
                print("Path found!")
                return False


drive_a_path()
