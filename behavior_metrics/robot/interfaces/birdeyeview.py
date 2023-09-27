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
            render_lanes_on_junctions=True,
            crop_type=BirdViewCropType.FRONT_AREA_ONLY
        )

    def getImage(self, vehicle):
        try:
            birdview = self.birdview_producer.produce(
                agent_vehicle=vehicle  # carla.Actor (spawned vehicle)
            )
        except Exception as ex:
            print(ex)
        # Mask to RGB image
        image = BirdViewProducer.as_rgb(birdview)
        return image
