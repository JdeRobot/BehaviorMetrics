from sensors import Sensors
import yaml

if __name__ == '__main__':

    with open('../driver.yml') as file:
        cfg = yaml.safe_load(file)

    
    sensors_cfg = cfg['Behaviors']['Robots']['Robot_0']['Sensors']  

    sensors = Sensors(sensors_cfg)

    print()