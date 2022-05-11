---
title: Circuit and weather selection
layout: posts
permalink: /documentation/circuit_and_weather/

collection: posts

classes: wide

sidebar:
  nav: "docs"
---

## Available circuits

Behavior Metrics has a wide range of circuits with different shapes. The basic version of the circuits comes with a red line 
on the middle of the road along the circuit. Additionally, there are variations of these circuits that include: no red line, white road, 
no red line and no walls, no walls and white road without red line.


| CIRCUITS           |            |          |  | | | | |
|                   | ROAD COLOR | Grey     | Grey       | Grey    | Grey     | White    |White    |
|                   | LINE       | Red line | white line | no line | no line  | red line | no line |
|                   | WALLS      | yes      | yes        | yes      | no      | yes      | yes |
|-------------------|------------|----------|----|--|--|--|--|
| Simple circuit    |           | YES      |  YES  | YES | YES| YES |YES |
| Many curves       |          | YES      |  YES  | YES | YES| YES |YES |
| Montmeló          |           | YES     |  YES  | YES | YES| YES |YES |
| Extended simple   |           | YES      |     | | | | | |
| Monacó            |           | YES        |     | | | | | |
| Montreal          |           | YES      |   | YES | YES|  | | |
| Nurburgring       |           | YES      |    | YES | YES|  | | |

## Weather conditions

Different weather conditions can be selected for a more realistic simulation.
The weather conditions can be selected adding them to the configuration file in the field `ImageTranform`

**Available options are:**

* Rain (`rain`)
* Night (`night`)
* Shadow (`shadow`)
* Snow (`snow`)
* Fog (`fog`)
* Sun flare (`sunflare`)


