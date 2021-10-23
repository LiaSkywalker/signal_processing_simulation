import nidaqmx
import nidaqmx.system


if __name__=="__main__":
    system = nidaqmx.system.System.local()
    for dev in system.devices:
        print(dev)
