---
permalink: /install/

title: "Installation and use"

sidebar:
  nav: "docs"
---

This application depends on dome third party libraries, most of them are included in the requirements file. To install them just type the following:

```bash
pip install -r requirements.txt
```

However since this project is built under python 2.7 the Qt5 dependencies cannot be installed via pip (it's included in the pip3 repositories). So it's needed to install the Qt5 dependencies through apt as follows:

```bash
INSTALL QT5 and PYQT5
```

[WIP]