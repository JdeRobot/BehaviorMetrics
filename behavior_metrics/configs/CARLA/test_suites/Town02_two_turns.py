TEST_ROUTES = [
    {
        "map": "Town02",
        "start": "-3.68, -288.22, 0.5, 0.0, 0.0, 90.0",
        "end": "41.39, -212.98, 0.5, 0.0, 0.0, -90.0",
        "distance": 158,
        "commands": ["Right", "Right"]
    },
    {
        "map": "Town02",
        "start":"136.11, -302.57, 0.5, 0.0, 0.0, 180.0",
        "end":"45.89, -216.26, 0.5, 0.0, 0.0, 90.0",
        "distance": 174,
        "commands": ["Right", "Straight"]
    },
    {
        "map": "Town02",
        "start": "45.24, -225.59, 0.5, 0.0, 0.0, 90",
        "end": "-7.53, -270.73, 0.5, 0.0, 0.0, -90.0",
        "distance": 166,
        "commands": ["Left", "Left"]
    },
    {
        "map": "Town02",
        "start": "41.39, -203.90, 0.5, 0.0, 0.0, -90.0",
        "end": "136.11, -306.42, 0.5, 0.0, 0.0, 0.0",
        "distance": 193,
        "commands": ["Straight", "Left"]
    },
    {
        "map": "Town02",
        "start": "59.71, -241.28, 0.5, 0.0, 0.0, 0.0",
        "end": "166.91, -191.77, 0.5, 0.0, 0.0, 0.0",
        "distance": 148,
        "commands": ["Left", "Right"]
    },
    {
        "map": "Town02",
        "start": "165.09, -187.12, 0.5, 0.0, 0.0, 180.0",
        "end": "71.04, -237.43, 0.5, 0.0, 0.0, 180.0",
        "distance": 136,
        "commands": ["Left", "Right"]
    },
    {
        "map": "Town02",
        "start": "59.60, -306.42, 0.5, 0.0, 0.0, 0.0",
        "end": "165.09, -187.12, 0.5, 0.0, 0.0, 180.0",
        "distance": 272,
        "commands": ["Straight", "Left"]
    },
    {
        "map": "Town02",
        "start": "166.91, -191.77, 0.5, 0.0, 0.0, 0.0",
        "end": "71.53, -302.57, 0.5, 0.0, 0.0, 180.0",
        "distance": 245,
        "commands": ["Right", "Straight"]
    },
    {
        "map": "Town02",
        "start": "92.34, -187.92, 0.5, 0.0, 0.0, 180.0",
        "end": "84.41, -109.30, 0.5, 0.0, 0.0, 0.0",
        "distance": 255,
        "commands": ["Straight", "Right"]
    },
    {
        "map": "Town02",
        "start": "84.41, -105.55, 0.5, 0.0, 0.0, 180.0",
        "end": "92.34, -191.77, 0.5, 0.0, 0.0, 0.0",
        "distance": 269,
        "commands": ["Left", "Straight"]
    },
    {
        "map": "Town02",
        "start": "135.88, -226.05, 0.5, 0.0, 0.0, 90.0",
        "end": "193.78, -121.21, 0.5, 0.0, 0.0, 90.0",
        "distance": 155,
        "commands": ["Right", "Left"]
    },
    {
        "map": "Town02",
        "start": "189.93, -121.21, 0.5, 0.0, 0.0, -90.0",
        "end": "132.03, -211.0, 0.5, 0.0, 0.0, -90.0",
        "distance": 140,
        "commands": ["Right", "Left"]
    }
]

if __name__ == "__main__":
    # Quick test to see if the routes are loaded correctly
    for route in TEST_ROUTES:
        print(route)