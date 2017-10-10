from multiprocessing import Process, Value
import time
import datetime

def currtime(n,start): #function running in backround keeping track of time
    while True:
        n.value = time.time() -start
        # n.value += 1
        time.sleep(1)

if __name__ == '__main__':
    start = time.time()
    n = Value('f', 0)
    p = Process(target=currtime, args=(n,start))
    p.start()
    while True:
        time.sleep(3)
        print(n.value)
