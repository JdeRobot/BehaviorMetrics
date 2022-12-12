import carla
from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions

class BirdEyeView:

    def __init__(self):
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        self.birdview_producer = BirdViewProducer(
            client,  # carla.Client
            target_size=PixelDimensions(width=100, height=300),
            pixels_per_meter=10,
            crop_type=BirdViewCropType.FRONT_AREA_ONLY
        )

    def getImage(self, vehicle):
        birdview = self.birdview_producer.produce(
            agent_vehicle=vehicle  # carla.Actor (spawned vehicle)
        )
        image = BirdViewProducer.as_rgb(birdview)
        return image
