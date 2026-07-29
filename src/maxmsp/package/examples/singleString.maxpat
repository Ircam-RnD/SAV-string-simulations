{
    "patcher": {
        "fileversion": 1,
        "appversion": {
            "major": 9,
            "minor": 1,
            "revision": 4,
            "architecture": "x64",
            "modernui": 1
        },
        "classnamespace": "box",
        "rect": [ 34.0, 95.0, 1444.0, 853.0 ],
        "openinpresentation": 1,
        "boxes": [
            {
                "box": {
                    "id": "obj-37",
                    "local": 1,
                    "maxclass": "ezdac~",
                    "numinlets": 2,
                    "numoutlets": 0,
                    "patching_rect": [ 11.0, 703.5, 45.0, 45.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 198.0, 574.5632088184357, 45.0, 45.0 ]
                }
            },
            {
                "box": {
                    "id": "obj-161",
                    "maxclass": "multislider",
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "", "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 326.643666267395, 628.0, 420.0, 122.0 ],
                    "setminmax": [ -0.0010000000474974513, 0.0010000000474974513 ],
                    "size": 139
                }
            },
            {
                "box": {
                    "id": "obj-185",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 781.0, 613.0, 150.0, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 591.0, 427.20114612579346, 197.5, 20.0 ],
                    "text": "NONLINEAR MODE"
                }
            },
            {
                "box": {
                    "border": 3.0,
                    "id": "obj-184",
                    "maxclass": "live.line",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 183.0, 679.0, 5.0, 100.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 591.0, 447.20114612579346, 233.0, 5.0 ]
                }
            },
            {
                "box": {
                    "id": "obj-182",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 781.0, 770.0, 150.0, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 608.0, 578.2011461257935, 150.0, 20.0 ],
                    "text": "4. Contact + Geom"
                }
            },
            {
                "box": {
                    "id": "obj-180",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 781.0, 748.0, 150.0, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 608.0, 556.2011461257935, 150.0, 20.0 ],
                    "text": "3. Contact"
                }
            },
            {
                "box": {
                    "id": "obj-179",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 781.0, 726.0, 150.0, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 608.0, 536.2011461257935, 150.0, 20.0 ],
                    "text": "2. Geom"
                }
            },
            {
                "box": {
                    "id": "obj-177",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 781.0, 703.0, 150.0, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 608.0, 514.2011461257935, 150.0, 20.0 ],
                    "text": "1. KC"
                }
            },
            {
                "box": {
                    "id": "obj-175",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 781.0, 680.0, 150.0, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 608.0, 492.20114612579346, 150.0, 20.0 ],
                    "text": "0. Linear"
                }
            },
            {
                "box": {
                    "attr": "nonlinear mode",
                    "id": "obj-159",
                    "maxclass": "attrui",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 781.0, 649.0, 150.0, 22.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 612.0, 463.20114612579346, 203.0, 22.0 ],
                    "text_width": 118.0
                }
            },
            {
                "box": {
                    "id": "obj-150",
                    "maxclass": "scope~",
                    "numinlets": 2,
                    "numoutlets": 0,
                    "patching_rect": [ 67.0, 796.0, 130.0, 130.0 ]
                }
            },
            {
                "box": {
                    "id": "obj-173",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 101.0, 563.0, 50.0, 22.0 ],
                    "text": "s outdis"
                }
            },
            {
                "box": {
                    "id": "obj-155",
                    "maxclass": "newobj",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "int" ],
                    "patching_rect": [ 1126.0, 258.0, 31.0, 22.0 ],
                    "text": "int 1"
                }
            },
            {
                "box": {
                    "id": "obj-154",
                    "maxclass": "newobj",
                    "numinlets": 2,
                    "numoutlets": 2,
                    "outlettype": [ "bang", "" ],
                    "patching_rect": [ 1126.0, 234.0, 50.0, 22.0 ],
                    "text": "select 6"
                }
            },
            {
                "box": {
                    "id": "obj-152",
                    "maxclass": "button",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "bang" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 1096.0, 247.27585792541504, 24.0, 24.0 ]
                }
            },
            {
                "box": {
                    "id": "obj-149",
                    "maxclass": "newobj",
                    "numinlets": 3,
                    "numoutlets": 1,
                    "outlettype": [ "signal" ],
                    "patching_rect": [ 126.0, 601.0, 68.0, 22.0 ],
                    "text": "min.xfade~"
                }
            },
            {
                "box": {
                    "id": "obj-148",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 284.0, 448.5, 69.0, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 450.0, 202.0, 69.0, 20.0 ],
                    "text": "Dry/Wet"
                }
            },
            {
                "box": {
                    "id": "obj-147",
                    "maxclass": "live.dial",
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "", "float" ],
                    "parameter_enable": 1,
                    "patching_rect": [ 250.0, 440.0, 27.0, 37.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 527.0, 193.5, 27.0, 37.0 ],
                    "saved_attribute_attributes": {
                        "valueof": {
                            "parameter_initial": [ 0.0 ],
                            "parameter_initial_enable": 1,
                            "parameter_linknames": 1,
                            "parameter_longname": "Dry/Wet",
                            "parameter_mmax": 1.0,
                            "parameter_modmode": 3,
                            "parameter_shortname": "speed",
                            "parameter_type": 0,
                            "parameter_unitstyle": 1
                        }
                    },
                    "showname": 0,
                    "varname": "Dry/Wet"
                }
            },
            {
                "box": {
                    "id": "obj-142",
                    "maxclass": "newobj",
                    "numinlets": 3,
                    "numoutlets": 1,
                    "outlettype": [ "signal" ],
                    "patching_rect": [ 11.0, 602.0, 68.0, 22.0 ],
                    "text": "min.xfade~"
                }
            },
            {
                "box": {
                    "id": "obj-139",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 3,
                    "outlettype": [ "", "", "" ],
                    "patching_rect": [ 121.0, 347.0, 68.0, 22.0 ],
                    "restore": [ 0.324583033035046 ],
                    "saved_object_attributes": {
                        "parameter_enable": 0,
                        "parameter_mappable": 0
                    },
                    "text": "pattr ingain",
                    "varname": "ingain"
                }
            },
            {
                "box": {
                    "id": "obj-138",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 955.0, 406.0, 150.0, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 660.5, 289.27586126327515, 150.0, 20.0 ],
                    "text": "Resonator"
                }
            },
            {
                "box": {
                    "id": "obj-136",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 955.0, 386.0, 150.0, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 660.5, 263.98850536346436, 150.0, 20.0 ],
                    "text": "Sawtooth sweep"
                }
            },
            {
                "box": {
                    "id": "obj-123",
                    "linecount": 2,
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 955.0, 353.0, 150.0, 33.0 ],
                    "presentation": 1,
                    "presentation_linecount": 2,
                    "presentation_rect": [ 660.5, 228.0, 150.0, 33.0 ],
                    "text": "Phasing effect, faster impacts, pluck"
                }
            },
            {
                "box": {
                    "id": "obj-122",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 955.0, 333.0, 150.0, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 660.0, 206.7554030418396, 150.0, 20.0 ],
                    "text": "Low string, hinarmonicity"
                }
            },
            {
                "box": {
                    "id": "obj-121",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 955.0, 313.0, 150.0, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 660.5, 177.0, 150.0, 20.0 ],
                    "text": "Low string"
                }
            },
            {
                "box": {
                    "id": "obj-120",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 955.0, 293.0, 150.0, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 660.5, 147.1428571428571, 150.0, 20.0 ],
                    "text": "High string"
                }
            },
            {
                "box": {
                    "format": 6,
                    "id": "obj-140",
                    "maxclass": "flonum",
                    "maximum": 1000.0,
                    "minimum": 0.0,
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "", "bang" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 561.1099406480789, 338.0, 50.0, 22.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 453.31033849716187, 509.79884219169617, 50.0, 22.0 ],
                    "varname": "number[1]"
                }
            },
            {
                "box": {
                    "id": "obj-135",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 3,
                    "outlettype": [ "", "", "" ],
                    "patching_rect": [ 559.6099406480789, 309.0, 40.0, 22.0 ],
                    "restore": [ 12.0 ],
                    "saved_object_attributes": {
                        "parameter_enable": 0,
                        "parameter_mappable": 0
                    },
                    "text": "pattr",
                    "varname": "u749001509"
                }
            },
            {
                "box": {
                    "id": "obj-134",
                    "linecount": 2,
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 3,
                    "outlettype": [ "", "", "" ],
                    "patching_rect": [ 556.7376002669334, 248.27585792541504, 58.37234038114548, 35.0 ],
                    "restore": [ "stability condition setting", 0.19970000000000007 ],
                    "saved_object_attributes": {
                        "parameter_enable": 0,
                        "parameter_mappable": 0
                    },
                    "text": "pattr kappa",
                    "varname": "kappa"
                }
            },
            {
                "box": {
                    "id": "obj-132",
                    "linecount": 2,
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 3,
                    "outlettype": [ "", "", "" ],
                    "patching_rect": [ 556.4876002669334, 211.89080095291138, 58.87234038114548, 35.0 ],
                    "restore": [ "regularisation parameter", 1000.0 ],
                    "saved_object_attributes": {
                        "parameter_enable": 0,
                        "parameter_mappable": 0
                    },
                    "text": "pattr lambda0",
                    "varname": "lambda0"
                }
            },
            {
                "box": {
                    "id": "obj-131",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 3,
                    "outlettype": [ "", "", "" ],
                    "patching_rect": [ 556.6099406480789, 185.86450910568237, 66.0, 22.0 ],
                    "restore": [ "second decay time", 6.0 ],
                    "saved_object_attributes": {
                        "parameter_enable": 0,
                        "parameter_mappable": 0
                    },
                    "text": "pattr t60_1",
                    "varname": "t60_1"
                }
            },
            {
                "box": {
                    "id": "obj-129",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 3,
                    "outlettype": [ "", "", "" ],
                    "patching_rect": [ 556.6099406480789, 153.24039494991302, 66.0, 22.0 ],
                    "restore": [ "second decay frequency", 1000.0 ],
                    "saved_object_attributes": {
                        "parameter_enable": 0,
                        "parameter_mappable": 0
                    },
                    "text": "pattr f60_1",
                    "varname": "f60_1"
                }
            },
            {
                "box": {
                    "id": "obj-127",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 3,
                    "outlettype": [ "", "", "" ],
                    "patching_rect": [ 556.6099406480789, 121.32550066709518, 66.0, 22.0 ],
                    "restore": [ "first decay time", 12.0 ],
                    "saved_object_attributes": {
                        "parameter_enable": 0,
                        "parameter_mappable": 0
                    },
                    "text": "pattr t60_0",
                    "varname": "t60_0"
                }
            },
            {
                "box": {
                    "id": "obj-126",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 3,
                    "outlettype": [ "", "", "" ],
                    "patching_rect": [ 556.6099406480789, 88.70138651132584, 66.0, 22.0 ],
                    "restore": [ "first decay frequency", 0.0 ],
                    "saved_object_attributes": {
                        "parameter_enable": 0,
                        "parameter_mappable": 0
                    },
                    "text": "pattr f60_0",
                    "varname": "f60_0"
                }
            },
            {
                "box": {
                    "id": "obj-125",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 3,
                    "outlettype": [ "", "", "" ],
                    "patching_rect": [ 556.6099406480789, 56.786492228507996, 59.0, 22.0 ],
                    "restore": [ "beta", 0.00016 ],
                    "saved_object_attributes": {
                        "parameter_enable": 0,
                        "parameter_mappable": 0
                    },
                    "text": "pattr beta",
                    "varname": "beta"
                }
            },
            {
                "box": {
                    "id": "obj-124",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 3,
                    "outlettype": [ "", "", "" ],
                    "patching_rect": [ 556.6099406480789, 25.580817818641663, 46.0, 22.0 ],
                    "restore": [ "fundamental frequency", 51.9 ],
                    "saved_object_attributes": {
                        "parameter_enable": 0,
                        "parameter_mappable": 0
                    },
                    "text": "pattr f0",
                    "varname": "f0"
                }
            },
            {
                "box": {
                    "id": "obj-72",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 4,
                    "outlettype": [ "", "", "", "" ],
                    "patching_rect": [ 1183.0, 228.33907675743103, 56.0, 22.0 ],
                    "restore": {
                        "Base excitation position": [ 0.3 ],
                        "Base left listening": [ 0.3 ],
                        "Base right listening": [ 0.638582677165353 ],
                        "Dry/Wet": [ 1.0 ],
                        "Excitation LF0 amplitude": [ 0.0 ],
                        "Excitation LF0 speed": [ 0.572440944881889 ],
                        "Impulse Width": [ 1.892913385826754 ],
                        "Left LF0 amplitude": [ 0.0 ],
                        "Left LF0 speed": [ 0.493700787401575 ],
                        "Right LF0 speed": [ 1.359842519685048 ],
                        "Right LFO amplitude": [ 0.0 ],
                        "Saw frequency": [ 0.0 ],
                        "Vibrato Amplitude": [ 0.0 ],
                        "Vibrato Speed": [ 8.051181102362214 ],
                        "button": [ 0.0 ],
                        "live.gain~": [ -26.432031718167387 ],
                        "number": [ 2000 ],
                        "toggleImpulse": [ 1 ],
                        "togglePluck": [ 1 ],
                        "toggleSaw": [ 0 ],
                        "umenu": [ 0 ]
                    },
                    "text": "autopattr",
                    "varname": "u560004266"
                }
            },
            {
                "box": {
                    "id": "obj-32",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 1071.0, 190.1091923713684, 92.0, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 610.0, 121.70494973659515, 150.0, 20.0 ],
                    "text": "Recall presets"
                }
            },
            {
                "box": {
                    "id": "obj-181",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "bang" ],
                    "patching_rect": [ 984.0, 119.25, 58.0, 22.0 ],
                    "text": "loadbang"
                }
            },
            {
                "box": {
                    "id": "obj-144",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 999.0, 36.25, 150.0, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 610.0, 66.70494973659515, 150.0, 20.0 ],
                    "text": "Store a preset at index"
                }
            },
            {
                "box": {
                    "id": "obj-143",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 942.0, 10.25, 150.0, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 614.0, 16.704949736595154, 150.0, 20.0 ],
                    "text": "PRESETS"
                }
            },
            {
                "box": {
                    "id": "obj-141",
                    "linecount": 2,
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 1183.0, 30.25, 154.0, 33.0 ],
                    "presentation": 1,
                    "presentation_linecount": 2,
                    "presentation_rect": [ 591.0, 321.85632038116455, 154.0, 33.0 ],
                    "text": "Write and read preset data to/from disk"
                }
            },
            {
                "box": {
                    "id": "obj-137",
                    "maxclass": "message",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 1184.0, 122.25, 33.0, 22.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 651.0, 357.85632038116455, 33.0, 22.0 ],
                    "text": "read"
                }
            },
            {
                "box": {
                    "id": "obj-133",
                    "maxclass": "message",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 958.0, 35.25, 35.0, 22.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 613.0, 94.70494973659515, 35.0, 22.0 ],
                    "text": "store"
                }
            },
            {
                "box": {
                    "id": "obj-130",
                    "maxclass": "number",
                    "maximum": 8,
                    "minimum": 1,
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "", "bang" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 988.0, 62.25, 50.0, 22.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 660.0, 94.70494973659515, 50.0, 22.0 ]
                }
            },
            {
                "box": {
                    "id": "obj-128",
                    "maxclass": "newobj",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 958.0, 92.25, 49.0, 22.0 ],
                    "text": "pack s i"
                }
            },
            {
                "box": {
                    "id": "obj-115",
                    "maxclass": "message",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 1183.0, 96.25, 89.0, 22.0 ],
                    "text": "storagewindow"
                }
            },
            {
                "box": {
                    "id": "obj-47",
                    "maxclass": "message",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 1183.0, 69.25, 34.0, 22.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 596.0, 357.85632038116455, 34.0, 22.0 ],
                    "text": "write"
                }
            },
            {
                "box": {
                    "autorestore": "singleString.json",
                    "id": "obj-50",
                    "linecount": 3,
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 1183.0, 150.25, 99.0, 49.0 ],
                    "saved_object_attributes": {
                        "client_rect": [ 1057, 45, 1440, 407 ],
                        "parameter_enable": 0,
                        "parameter_mappable": 0,
                        "storage_rect": [ 343, 309, 1338, 857 ]
                    },
                    "text": "pattrstorage presetmultiple @savemode 0",
                    "varname": "presetmultiple"
                }
            },
            {
                "box": {
                    "id": "obj-55",
                    "maxclass": "message",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "patching_rect": [ 958.0, 145.25, 151.0, 22.0 ],
                    "text": "pattrstorage presetmultiple"
                }
            },
            {
                "box": {
                    "bubblesize": 24,
                    "id": "obj-56",
                    "maxclass": "preset",
                    "numinlets": 1,
                    "numoutlets": 5,
                    "outlettype": [ "preset", "int", "preset", "int", "" ],
                    "patching_rect": [ 958.0, 183.0, 103.0, 92.0 ],
                    "pattrstorage": "presetmultiple",
                    "presentation": 1,
                    "presentation_rect": [ 613.0, 144.0, 35.0, 178.27586126327515 ]
                }
            },
            {
                "box": {
                    "angle": 270.0,
                    "bgcolor": [ 0.133674, 0.354866, 0.236742, 1.0 ],
                    "id": "obj-170",
                    "maxclass": "panel",
                    "mode": 0,
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 935.0, 6.0, 402.0, 282.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 591.0, 13.0, 289.0, 398.0 ],
                    "proportion": 0.5
                }
            },
            {
                "box": {
                    "id": "obj-114",
                    "linecount": 2,
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 378.8148078918457, 327.28735089302063, 150.0, 33.0 ],
                    "presentation": 1,
                    "presentation_linecount": 2,
                    "presentation_rect": [ 299.5, 504.29884219169617, 150.0, 33.0 ],
                    "text": "First decay time, may be modulated without reset"
                }
            },
            {
                "box": {
                    "id": "obj-29",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "bang" ],
                    "patching_rect": [ 557.1099406480789, 292.1985876560211, 58.0, 22.0 ],
                    "text": "loadbang"
                }
            },
            {
                "box": {
                    "border": 2.0,
                    "id": "obj-119",
                    "maxclass": "live.line",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 153.0, 649.0, 5.0, 100.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 373.0, 59.0, 5.0, 121.28571428571428 ]
                }
            },
            {
                "box": {
                    "border": 2.0,
                    "id": "obj-118",
                    "maxclass": "live.line",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 153.0, 649.0, 5.0, 100.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 203.07194864749908, 120.0, 5.0, 60.28571428571428 ]
                }
            },
            {
                "box": {
                    "border": 2.0,
                    "id": "obj-117",
                    "maxclass": "live.line",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 153.0, 649.0, 5.0, 100.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 203.07194864749908, 58.99280786514282, 5.0, 54.53129939734936 ]
                }
            },
            {
                "box": {
                    "id": "obj-116",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 192.0, 640.0, 74.04598021507263, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 349.31033849716187, 574.7126340866089, 77.0, 20.0 ],
                    "text": "Samplerate"
                }
            },
            {
                "box": {
                    "annotation": "Samplerate",
                    "blanksym": "",
                    "id": "obj-111",
                    "items": [ 44100, ",", 48000, ",", 88200, ",", 96000 ],
                    "maxclass": "umenu",
                    "numinlets": 1,
                    "numoutlets": 3,
                    "outlettype": [ "int", "", "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 185.0, 710.0, 100.0, 22.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 428.31033849716187, 574.7126340866089, 100.0, 22.0 ],
                    "varname": "umenu"
                }
            },
            {
                "box": {
                    "id": "obj-112",
                    "maxclass": "newobj",
                    "numinlets": 2,
                    "numoutlets": 2,
                    "outlettype": [ "", "int" ],
                    "patching_rect": [ 185.0, 670.0, 67.0, 22.0 ],
                    "text": "adstatus sr"
                }
            },
            {
                "box": {
                    "id": "obj-110",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 221.0, 259.0, 66.0, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 248.47122430801392, 118.70504021644592, 78.0, 20.0 ],
                    "text": "Frequency",
                    "textjustification": 1
                }
            },
            {
                "box": {
                    "id": "obj-109",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 273.0, 310.0, 66.0, 20.0 ],
                    "text": "On/Off",
                    "textjustification": 1
                }
            },
            {
                "box": {
                    "id": "obj-108",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 268.0, 235.0, 66.0, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 247.47122430801392, 65.0, 80.0, 20.0 ],
                    "text": "Sawtooth",
                    "textjustification": 1
                }
            },
            {
                "box": {
                    "id": "obj-39",
                    "maxclass": "newobj",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "signal" ],
                    "patching_rect": [ 216.0, 338.0, 37.0, 22.0 ],
                    "text": "saw~"
                }
            },
            {
                "box": {
                    "format": 6,
                    "id": "obj-38",
                    "maxclass": "flonum",
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "", "bang" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 216.0, 309.0, 50.0, 22.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 261.9712243080139, 157.25, 51.0, 22.0 ]
                }
            },
            {
                "box": {
                    "floatoutput": 1,
                    "id": "obj-106",
                    "maxclass": "slider",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 221.0, 283.0, 123.0, 18.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 235.97122430801392, 140.5, 103.0, 13.0 ],
                    "size": 1000.0,
                    "varname": "Saw frequency"
                }
            },
            {
                "box": {
                    "id": "obj-107",
                    "maxclass": "newobj",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "signal" ],
                    "patching_rect": [ 216.0, 369.0, 29.5, 22.0 ],
                    "text": "*~"
                }
            },
            {
                "box": {
                    "id": "obj-24",
                    "maxclass": "toggle",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "int" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 273.0, 333.0, 24.0, 24.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 275.4712243080139, 83.61151385307312, 24.0, 24.0 ],
                    "varname": "toggleSaw"
                }
            },
            {
                "box": {
                    "id": "obj-105",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 216.0, 66.0, 101.5359696149826, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 379.31033557653427, 65.0, 198.0000058412552, 20.0 ],
                    "text": "Audio Input",
                    "textjustification": 1
                }
            },
            {
                "box": {
                    "clipheight": 45.0,
                    "data": {
                        "clips": [
                            {
                                "absolutepath": "/Users/risse/Projects/Travail ISMM/web-audio-samples/_site/sounds/fx/obama-oilspill.mp3",
                                "filename": "obama-oilspill.mp3",
                                "filekind": "audiofile",
                                "id": "u989003145",
                                "loop": 0,
                                "content_state": {                                }
                            },
                            {
                                "absolutepath": "/Users/risse/Projects/NonlinearString/SAV-string-simulations/src/maxmsp/package/media/woodbox.wav",
                                "filename": "woodbox.wav",
                                "filekind": "audiofile",
                                "id": "u197004245",
                                "loop": 0,
                                "content_state": {                                }
                            }
                        ]
                    },
                    "id": "obj-48",
                    "maxclass": "playlist~",
                    "mode": "basic",
                    "numinlets": 1,
                    "numoutlets": 5,
                    "outlettype": [ "signal", "signal", "signal", "", "dictionary" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 216.0, 95.0, 150.0, 92.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 379.31033849716187, 85.05746984481812, 198.0, 93.29495978355408 ],
                    "quality": "basic",
                    "saved_attribute_attributes": {
                        "candicane2": {
                            "expression": ""
                        },
                        "candicane3": {
                            "expression": ""
                        },
                        "candicane4": {
                            "expression": ""
                        },
                        "candicane5": {
                            "expression": ""
                        },
                        "candicane6": {
                            "expression": ""
                        },
                        "candicane7": {
                            "expression": ""
                        },
                        "candicane8": {
                            "expression": ""
                        }
                    }
                }
            },
            {
                "box": {
                    "border": 2.0,
                    "id": "obj-104",
                    "maxclass": "live.line",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 153.0, 649.0, 5.0, 100.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 9.23381707072258, 391.5747101306915, 285.0, 10.5 ]
                }
            },
            {
                "box": {
                    "border": 2.0,
                    "id": "obj-103",
                    "maxclass": "live.line",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 153.0, 649.0, 5.0, 100.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 8.084391802549362, 327.20689511299133, 285.0, 10.5 ]
                }
            },
            {
                "box": {
                    "border": 3.0,
                    "id": "obj-102",
                    "maxclass": "live.line",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 153.0, 649.0, 5.0, 100.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 9.23381707072258, 257.09195375442505, 265.0, 9.0 ]
                }
            },
            {
                "box": {
                    "id": "obj-91",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 634.4827480316162, 399.45401644706726, 130.0, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 9.23381707072258, 393.87356066703796, 130.0, 20.0 ],
                    "text": "Right listening position"
                }
            },
            {
                "box": {
                    "id": "obj-92",
                    "maxclass": "newobj",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "signal" ],
                    "patching_rect": [ 634.4827480316162, 501.75286531448364, 29.5, 22.0 ],
                    "text": "+~"
                }
            },
            {
                "box": {
                    "id": "obj-93",
                    "maxclass": "newobj",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "signal" ],
                    "patching_rect": [ 712.643666267395, 501.75286531448364, 34.0, 22.0 ],
                    "text": "*~ 0."
                }
            },
            {
                "box": {
                    "id": "obj-94",
                    "maxclass": "newobj",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "signal" ],
                    "patching_rect": [ 712.643666267395, 473.0172336101532, 43.0, 22.0 ],
                    "text": "cycle~"
                }
            },
            {
                "box": {
                    "id": "obj-95",
                    "maxclass": "newobj",
                    "numinlets": 3,
                    "numoutlets": 1,
                    "outlettype": [ "signal" ],
                    "patching_rect": [ 634.4827480316162, 531.6379222869873, 60.0, 22.0 ],
                    "text": "clip~ 0. 1."
                }
            },
            {
                "box": {
                    "id": "obj-96",
                    "linecount": 2,
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 832.1838941574097, 429.3390734195709, 69.0, 33.0 ],
                    "presentation": 1,
                    "presentation_linecount": 2,
                    "presentation_rect": [ 220.72806641459465, 418.01149129867554, 69.0, 33.0 ],
                    "text": "LF0\nAmplitude"
                }
            },
            {
                "box": {
                    "id": "obj-97",
                    "maxclass": "live.dial",
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "", "float" ],
                    "parameter_enable": 1,
                    "patching_rect": [ 799.9999866485596, 427.0402228832245, 27.0, 37.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 193.14185997843742, 416.8620660305023, 27.0, 37.0 ],
                    "saved_attribute_attributes": {
                        "valueof": {
                            "parameter_initial": [ 0.0 ],
                            "parameter_initial_enable": 1,
                            "parameter_linknames": 1,
                            "parameter_longname": "Right LFO amplitude",
                            "parameter_mmax": 1.0,
                            "parameter_modmode": 3,
                            "parameter_shortname": "speed",
                            "parameter_type": 0,
                            "parameter_unitstyle": 1
                        }
                    },
                    "showname": 0,
                    "varname": "Right LFO amplitude"
                }
            },
            {
                "box": {
                    "id": "obj-98",
                    "linecount": 2,
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 743.6781485080719, 429.3390734195709, 49.0, 33.0 ],
                    "presentation": 1,
                    "presentation_linecount": 2,
                    "presentation_rect": [ 134.52117130160332, 418.01149129867554, 49.0, 33.0 ],
                    "text": "LFO\nSpeed"
                }
            },
            {
                "box": {
                    "id": "obj-99",
                    "maxclass": "live.dial",
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "", "float" ],
                    "parameter_enable": 1,
                    "patching_rect": [ 712.643666267395, 427.0402228832245, 27.0, 37.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 102.33726379275322, 416.8620660305023, 27.0, 37.0 ],
                    "saved_attribute_attributes": {
                        "valueof": {
                            "parameter_initial": [ 0.1 ],
                            "parameter_initial_enable": 1,
                            "parameter_linknames": 1,
                            "parameter_longname": "Right LF0 speed",
                            "parameter_mmax": 10.0,
                            "parameter_modmode": 3,
                            "parameter_shortname": "speed",
                            "parameter_type": 0,
                            "parameter_unitstyle": 1
                        }
                    },
                    "showname": 0,
                    "varname": "Right LF0 speed"
                }
            },
            {
                "box": {
                    "id": "obj-100",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 666.6666555404663, 435.086199760437, 37.0, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 49.4637014567852, 424.90804290771484, 37.0, 20.0 ],
                    "text": "Base"
                }
            },
            {
                "box": {
                    "id": "obj-101",
                    "maxclass": "live.dial",
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "", "float" ],
                    "parameter_enable": 1,
                    "patching_rect": [ 634.4827480316162, 425.89079761505127, 27.0, 37.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 13.831518143415451, 416.8620660305023, 27.0, 37.0 ],
                    "saved_attribute_attributes": {
                        "valueof": {
                            "parameter_initial": [ 0.3 ],
                            "parameter_initial_enable": 1,
                            "parameter_linknames": 1,
                            "parameter_longname": "Base right listening",
                            "parameter_mmax": 1.0,
                            "parameter_modmode": 3,
                            "parameter_shortname": "speed",
                            "parameter_type": 0,
                            "parameter_unitstyle": 1
                        }
                    },
                    "showname": 0,
                    "varname": "Base right listening"
                }
            },
            {
                "box": {
                    "id": "obj-90",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 634.4827480316162, 229.33907675743103, 130.0, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 9.23381707072258, 328.35632038116455, 130.0, 20.0 ],
                    "text": "Left listening position"
                }
            },
            {
                "box": {
                    "id": "obj-80",
                    "maxclass": "newobj",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "signal" ],
                    "patching_rect": [ 634.4827480316162, 332.78735089302063, 29.5, 22.0 ],
                    "text": "+~"
                }
            },
            {
                "box": {
                    "id": "obj-81",
                    "maxclass": "newobj",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "signal" ],
                    "patching_rect": [ 712.643666267395, 332.78735089302063, 34.0, 22.0 ],
                    "text": "*~ 0."
                }
            },
            {
                "box": {
                    "id": "obj-82",
                    "maxclass": "newobj",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "signal" ],
                    "patching_rect": [ 712.643666267395, 302.90229392051697, 43.0, 22.0 ],
                    "text": "cycle~"
                }
            },
            {
                "box": {
                    "id": "obj-83",
                    "maxclass": "newobj",
                    "numinlets": 3,
                    "numoutlets": 1,
                    "outlettype": [ "signal" ],
                    "patching_rect": [ 634.4827480316162, 361.5229825973511, 60.0, 22.0 ],
                    "text": "clip~ 0. 1."
                }
            },
            {
                "box": {
                    "id": "obj-84",
                    "linecount": 2,
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 832.1838941574097, 259.2241337299347, 69.0, 33.0 ],
                    "presentation": 1,
                    "presentation_linecount": 2,
                    "presentation_rect": [ 220.72806641459465, 355.9425268173218, 69.0, 33.0 ],
                    "text": "LF0\nAmplitude"
                }
            },
            {
                "box": {
                    "id": "obj-85",
                    "maxclass": "live.dial",
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "", "float" ],
                    "parameter_enable": 1,
                    "patching_rect": [ 799.9999866485596, 256.92528319358826, 27.0, 37.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 193.14185997843742, 353.64367628097534, 27.0, 37.0 ],
                    "saved_attribute_attributes": {
                        "valueof": {
                            "parameter_initial": [ 0.0 ],
                            "parameter_initial_enable": 1,
                            "parameter_linknames": 1,
                            "parameter_longname": "Left LF0 amplitude",
                            "parameter_mmax": 1.0,
                            "parameter_modmode": 3,
                            "parameter_shortname": "speed",
                            "parameter_type": 0,
                            "parameter_unitstyle": 1
                        }
                    },
                    "showname": 0,
                    "varname": "Left LF0 amplitude"
                }
            },
            {
                "box": {
                    "id": "obj-86",
                    "linecount": 2,
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 743.6781485080719, 259.2241337299347, 49.0, 33.0 ],
                    "presentation": 1,
                    "presentation_linecount": 2,
                    "presentation_rect": [ 134.52117130160332, 355.9425268173218, 49.0, 33.0 ],
                    "text": "LFO\nSpeed"
                }
            },
            {
                "box": {
                    "id": "obj-87",
                    "maxclass": "live.dial",
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "", "float" ],
                    "parameter_enable": 1,
                    "patching_rect": [ 712.643666267395, 256.92528319358826, 27.0, 37.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 102.33726379275322, 353.64367628097534, 27.0, 37.0 ],
                    "saved_attribute_attributes": {
                        "valueof": {
                            "parameter_initial": [ 0.1 ],
                            "parameter_initial_enable": 1,
                            "parameter_linknames": 1,
                            "parameter_longname": "Left LF0 speed",
                            "parameter_mmax": 10.0,
                            "parameter_modmode": 3,
                            "parameter_shortname": "speed",
                            "parameter_type": 0,
                            "parameter_unitstyle": 1
                        }
                    },
                    "showname": 0,
                    "varname": "Left LF0 speed"
                }
            },
            {
                "box": {
                    "id": "obj-88",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 666.6666555404663, 264.9712600708008, 37.0, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 49.4637014567852, 362.8390784263611, 37.0, 20.0 ],
                    "text": "Base"
                }
            },
            {
                "box": {
                    "id": "obj-89",
                    "maxclass": "live.dial",
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "", "float" ],
                    "parameter_enable": 1,
                    "patching_rect": [ 634.4827480316162, 255.77585792541504, 27.0, 37.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 13.831518143415451, 353.64367628097534, 27.0, 37.0 ],
                    "saved_attribute_attributes": {
                        "valueof": {
                            "parameter_initial": [ 0.3 ],
                            "parameter_initial_enable": 1,
                            "parameter_linknames": 1,
                            "parameter_longname": "Base left listening",
                            "parameter_mmax": 1.0,
                            "parameter_modmode": 3,
                            "parameter_shortname": "speed",
                            "parameter_type": 0,
                            "parameter_unitstyle": 1
                        }
                    },
                    "showname": 0,
                    "varname": "Base left listening"
                }
            },
            {
                "box": {
                    "id": "obj-79",
                    "maxclass": "newobj",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "signal" ],
                    "patching_rect": [ 634.4827480316162, 159.22413539886475, 29.5, 22.0 ],
                    "text": "+~"
                }
            },
            {
                "box": {
                    "id": "obj-78",
                    "maxclass": "newobj",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "signal" ],
                    "patching_rect": [ 712.643666267395, 159.22413539886475, 34.0, 22.0 ],
                    "text": "*~ 0."
                }
            },
            {
                "box": {
                    "id": "obj-77",
                    "maxclass": "newobj",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "signal" ],
                    "patching_rect": [ 712.643666267395, 130.4885036945343, 43.0, 22.0 ],
                    "text": "cycle~"
                }
            },
            {
                "box": {
                    "id": "obj-73",
                    "maxclass": "newobj",
                    "numinlets": 3,
                    "numoutlets": 1,
                    "outlettype": [ "signal" ],
                    "patching_rect": [ 634.4827480316162, 189.1091923713684, 60.0, 22.0 ],
                    "text": "clip~ 0. 1."
                }
            },
            {
                "box": {
                    "id": "obj-70",
                    "linecount": 2,
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 832.1838941574097, 86.81034350395203, 69.0, 33.0 ],
                    "presentation": 1,
                    "presentation_linecount": 2,
                    "presentation_rect": [ 220.72806641459465, 289.27586126327515, 69.0, 33.0 ],
                    "text": "LF0\nAmplitude"
                }
            },
            {
                "box": {
                    "id": "obj-71",
                    "maxclass": "live.dial",
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "", "float" ],
                    "parameter_enable": 1,
                    "patching_rect": [ 799.9999866485596, 85.66091823577881, 27.0, 37.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 193.14185997843742, 286.9770107269287, 27.0, 37.0 ],
                    "saved_attribute_attributes": {
                        "valueof": {
                            "parameter_initial": [ 0.0 ],
                            "parameter_initial_enable": 1,
                            "parameter_linknames": 1,
                            "parameter_longname": "Excitation LF0 amplitude",
                            "parameter_mmax": 1.0,
                            "parameter_modmode": 3,
                            "parameter_shortname": "speed",
                            "parameter_type": 0,
                            "parameter_unitstyle": 1
                        }
                    },
                    "showname": 0,
                    "varname": "Excitation LF0 amplitude"
                }
            },
            {
                "box": {
                    "id": "obj-68",
                    "linecount": 2,
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 743.6781485080719, 86.81034350395203, 49.0, 33.0 ],
                    "presentation": 1,
                    "presentation_linecount": 2,
                    "presentation_rect": [ 134.52117130160332, 289.27586126327515, 49.0, 33.0 ],
                    "text": "LFO\nSpeed"
                }
            },
            {
                "box": {
                    "id": "obj-69",
                    "maxclass": "live.dial",
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "", "float" ],
                    "parameter_enable": 1,
                    "patching_rect": [ 712.643666267395, 84.51149296760559, 27.0, 37.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 102.33726379275322, 286.9770107269287, 27.0, 37.0 ],
                    "saved_attribute_attributes": {
                        "valueof": {
                            "parameter_initial": [ 0.1 ],
                            "parameter_initial_enable": 1,
                            "parameter_linknames": 1,
                            "parameter_longname": "Excitation LF0 speed",
                            "parameter_mmax": 10.0,
                            "parameter_modmode": 3,
                            "parameter_shortname": "speed",
                            "parameter_type": 0,
                            "parameter_unitstyle": 1
                        }
                    },
                    "showname": 0,
                    "varname": "Excitation LF0 speed"
                }
            },
            {
                "box": {
                    "id": "obj-67",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 666.6666555404663, 92.55746984481812, 37.0, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 49.4637014567852, 293.873562335968, 37.0, 20.0 ],
                    "text": "Base"
                }
            },
            {
                "box": {
                    "id": "obj-63",
                    "maxclass": "live.dial",
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "", "float" ],
                    "parameter_enable": 1,
                    "patching_rect": [ 634.4827480316162, 84.51149296760559, 27.0, 37.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 13.831518143415451, 286.9770107269287, 27.0, 37.0 ],
                    "saved_attribute_attributes": {
                        "valueof": {
                            "parameter_initial": [ 0.3 ],
                            "parameter_initial_enable": 1,
                            "parameter_linknames": 1,
                            "parameter_longname": "Base excitation position",
                            "parameter_mmax": 1.0,
                            "parameter_modmode": 3,
                            "parameter_shortname": "speed",
                            "parameter_type": 0,
                            "parameter_unitstyle": 1
                        }
                    },
                    "showname": 0,
                    "varname": "Base excitation position"
                }
            },
            {
                "box": {
                    "id": "obj-60",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 634.4827480316162, 51.178160190582275, 84.0, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 9.23381707072258, 263.98850536346436, 84.0, 20.0 ],
                    "text": "Excitation"
                }
            },
            {
                "box": {
                    "id": "obj-45",
                    "linecount": 2,
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 634.4827480316162, 12.097701072692871, 192.0, 33.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 9.23381707072258, 241.0, 255.0, 20.0 ],
                    "text": "EXCITATION AND LISTENING POSITIONS"
                }
            },
            {
                "box": {
                    "attr": "stability condition setting",
                    "id": "obj-19",
                    "maxclass": "attrui",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 386.5248307585716, 248.27585792541504, 168.0851098895073, 22.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 299.99999499320984, 438.70114612579346, 264.3678116798401, 22.0 ],
                    "text_width": 112.7659597992897,
                    "varname": "attrui[7]"
                }
            },
            {
                "box": {
                    "attr": "regularisation parameter",
                    "id": "obj-2",
                    "maxclass": "attrui",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 386.5248307585716, 218.39080095291138, 168.0851098895073, 22.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 299.99999499320984, 414.5632154941559, 264.3678116798401, 22.0 ],
                    "text_width": 112.7659597992897,
                    "varname": "attrui[6]"
                }
            },
            {
                "box": {
                    "border": 3.0,
                    "id": "obj-76",
                    "maxclass": "live.line",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 153.0, 649.0, 5.0, 100.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 10.0, 569.1264276504517, 571.6091856956482, 5.0 ]
                }
            },
            {
                "box": {
                    "id": "obj-75",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 72.0, 705.0, 60.43165683746338, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 10.34482741355896, 547.1264276504517, 63.30935478210449, 20.0 ],
                    "text": "OUTPUT"
                }
            },
            {
                "box": {
                    "id": "obj-66",
                    "maxclass": "newobj",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "signal" ],
                    "patching_rect": [ 481.0, 462.0, 31.0, 22.0 ],
                    "text": "sig~"
                }
            },
            {
                "box": {
                    "id": "obj-64",
                    "maxclass": "live.dial",
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "", "float" ],
                    "parameter_enable": 1,
                    "patching_rect": [ 481.0, 416.0, 27.0, 37.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 164.24084946513176, 495.79884219169617, 27.0, 37.0 ],
                    "saved_attribute_attributes": {
                        "valueof": {
                            "parameter_initial": [ 0 ],
                            "parameter_initial_enable": 1,
                            "parameter_linknames": 1,
                            "parameter_longname": "Vibrato Amplitude",
                            "parameter_mmax": 1000.0,
                            "parameter_modmode": 3,
                            "parameter_shortname": "amplitude",
                            "parameter_type": 0,
                            "parameter_unitstyle": 1
                        }
                    },
                    "showname": 0,
                    "varname": "Vibrato Amplitude"
                }
            },
            {
                "box": {
                    "border": 3.0,
                    "id": "obj-62",
                    "maxclass": "live.line",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 153.0, 649.0, 5.0, 100.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 9.195402145385742, 485.21838307380676, 284.03841492533684, 5.586206436157227 ]
                }
            },
            {
                "box": {
                    "id": "obj-61",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 414.0, 425.0, 47.48201608657837, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 74.10072207450867, 504.29884219169617, 46.0172073841095, 20.0 ],
                    "text": "Speed"
                }
            },
            {
                "box": {
                    "border": 2.0,
                    "id": "obj-58",
                    "maxclass": "live.line",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 153.0, 649.0, 5.0, 100.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 15.0, 115.5, 360.0, 9.5 ]
                }
            },
            {
                "box": {
                    "border": 3.0,
                    "id": "obj-57",
                    "maxclass": "live.line",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 153.0, 649.0, 5.0, 100.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 298.29885053634644, 258.24137902259827, 231.58044409751892, 9.028734922409058 ]
                }
            },
            {
                "box": {
                    "border": 2.0,
                    "id": "obj-54",
                    "maxclass": "live.line",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 153.0, 649.0, 5.0, 100.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 15.0, 181.0, 566.3165469169617, 5.0 ]
                }
            },
            {
                "box": {
                    "border": 3.0,
                    "id": "obj-53",
                    "maxclass": "live.line",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 153.0, 649.0, 5.0, 100.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 10.0, 29.0, 571.3165469169617, 7.0 ]
                }
            },
            {
                "box": {
                    "border": 2.0,
                    "id": "obj-52",
                    "maxclass": "live.line",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 153.0, 649.0, 5.0, 100.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 16.0, 58.99280786514282, 565.3165498375893, 6.007192134857178 ]
                }
            },
            {
                "box": {
                    "border": 2.0,
                    "id": "obj-51",
                    "maxclass": "live.line",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 153.0, 649.0, 5.0, 100.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 64.8938412219286, 58.99280786514282, 5.0, 54.53129939734936 ]
                }
            },
            {
                "box": {
                    "id": "obj-46",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 11.0, 9.0, 150.0, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 10.0, 11.0, 49.0, 20.0 ],
                    "text": "INPUT"
                }
            },
            {
                "box": {
                    "attr": "second decay time",
                    "id": "obj-27",
                    "maxclass": "attrui",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 386.5248307585716, 189.65516924858093, 181.50000500679016, 22.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 299.99999499320984, 389.2758595943451, 264.3678116798401, 22.0 ],
                    "text_width": 154.00000500679016,
                    "varname": "attrui[5]"
                }
            },
            {
                "box": {
                    "id": "obj-4",
                    "maxclass": "newobj",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "signal" ],
                    "patching_rect": [ 386.0, 495.0, 112.95682573318481, 22.0 ],
                    "text": "*~"
                }
            },
            {
                "box": {
                    "attr": "first decay time",
                    "id": "obj-18",
                    "maxclass": "attrui",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 386.5248307585716, 129.8850553035736, 168.0851098895073, 22.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 299.99999499320984, 342.14942359924316, 264.3678116798401, 22.0 ],
                    "text_width": 136.00000500679016,
                    "varname": "attrui[3]"
                }
            },
            {
                "box": {
                    "id": "obj-16",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 40.0, 349.0, 72.0, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 10.071942806243896, 181.29497051239014, 197.9411783516407, 20.0 ],
                    "text": "Input Gain",
                    "textjustification": 1
                }
            },
            {
                "box": {
                    "id": "obj-14",
                    "maxclass": "newobj",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "signal" ],
                    "patching_rect": [ 11.0, 433.0, 48.05035924911499, 22.0 ],
                    "text": "*~"
                }
            },
            {
                "box": {
                    "format": 6,
                    "id": "obj-5",
                    "maxclass": "flonum",
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "", "bang" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 40.0, 399.0, 50.0, 22.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 152.74084946513176, 205.7554030418396, 50.0, 22.0 ]
                }
            },
            {
                "box": {
                    "id": "obj-36",
                    "lastchannelcount": 0,
                    "maxclass": "live.gain~",
                    "numinlets": 2,
                    "numoutlets": 5,
                    "orientation": 1,
                    "outlettype": [ "signal", "signal", "", "float", "list" ],
                    "parameter_enable": 1,
                    "patching_rect": [ 11.0, 645.0, 136.0, 47.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 20.68965482711792, 573.5632088184357, 136.0, 47.0 ],
                    "saved_attribute_attributes": {
                        "valueof": {
                            "parameter_longname": "live.gain~",
                            "parameter_mmax": 6.0,
                            "parameter_mmin": -70.0,
                            "parameter_modmode": 3,
                            "parameter_shortname": "live.gain~",
                            "parameter_type": 0,
                            "parameter_unitstyle": 4
                        }
                    },
                    "varname": "live.gain~"
                }
            },
            {
                "box": {
                    "id": "obj-35",
                    "linecount": 2,
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 511.0, 418.0, 66.1870527267456, 33.0 ],
                    "presentation": 1,
                    "presentation_linecount": 2,
                    "presentation_rect": [ 194.12590643763542, 498.0976927280426, 65.0, 33.0 ],
                    "text": "Amplitude (cents)"
                }
            },
            {
                "box": {
                    "id": "obj-34",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 386.0, 383.0, 60.43165683746338, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 9.31654691696167, 463.21838307380676, 63.30935478210449, 20.0 ],
                    "text": "VIBRATO"
                }
            },
            {
                "box": {
                    "id": "obj-33",
                    "maxclass": "live.dial",
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "", "float" ],
                    "parameter_enable": 1,
                    "patching_rect": [ 386.0, 416.0, 27.0, 37.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 41.91681456565857, 495.1034400463104, 27.0, 37.0 ],
                    "saved_attribute_attributes": {
                        "valueof": {
                            "parameter_linknames": 1,
                            "parameter_longname": "Vibrato Speed",
                            "parameter_mmax": 10.0,
                            "parameter_mmin": 0.1,
                            "parameter_modmode": 3,
                            "parameter_shortname": "speed",
                            "parameter_type": 0,
                            "parameter_unitstyle": 1
                        }
                    },
                    "showname": 0,
                    "varname": "Vibrato Speed"
                }
            },
            {
                "box": {
                    "id": "obj-31",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 386.2068901062012, 11.494252681732178, 194.24461126327515, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 296.0, 241.0, 157.0, 20.0 ],
                    "text": "PHYSICAL PARAMETERS"
                }
            },
            {
                "box": {
                    "id": "obj-25",
                    "linecount": 3,
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 434.0459702014923, 283.90229392051697, 115.0, 47.0 ],
                    "presentation": 1,
                    "presentation_linecount": 2,
                    "presentation_rect": [ 331.5, 464.304589509964, 242.0, 33.0 ],
                    "text": "Update parameters -> needed to apply changes !!"
                }
            },
            {
                "box": {
                    "id": "obj-23",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 80.0, 240.0, 84.67985343933105, 20.0 ],
                    "presentation": 1,
                    "presentation_linecount": 2,
                    "presentation_rect": [ 40.31654691696167, 134.1428571428571, 54.0, 33.0 ],
                    "text": "Impulse width"
                }
            },
            {
                "box": {
                    "id": "obj-21",
                    "maxclass": "live.dial",
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "", "float" ],
                    "parameter_enable": 1,
                    "patching_rect": [ 52.0, 231.0, 27.0, 37.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 10.31654691696167, 132.1428571428571, 27.0, 37.0 ],
                    "saved_attribute_attributes": {
                        "valueof": {
                            "parameter_initial": [ 0.1 ],
                            "parameter_initial_enable": 1,
                            "parameter_linknames": 1,
                            "parameter_longname": "Impulse Width",
                            "parameter_mmax": 10.0,
                            "parameter_mmin": 0.1,
                            "parameter_modmode": 3,
                            "parameter_shortname": "Imp. Width",
                            "parameter_type": 0,
                            "parameter_unitstyle": 1
                        }
                    },
                    "showname": 0,
                    "varname": "Impulse Width"
                }
            },
            {
                "box": {
                    "id": "obj-20",
                    "maxclass": "newobj",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "signal" ],
                    "patching_rect": [ 386.0, 462.0, 53.0, 22.0 ],
                    "text": "cycle~ 1"
                }
            },
            {
                "box": {
                    "id": "obj-30",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 118.0, 277.0, 74.0, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 129.31654691696167, 140.1428571428571, 75.0, 20.0 ],
                    "text": "Pluck/struck"
                }
            },
            {
                "box": {
                    "id": "obj-17",
                    "maxclass": "toggle",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "int" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 93.0, 275.0, 24.0, 24.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 105.81654691696167, 138.1428571428571, 24.0, 24.0 ],
                    "varname": "togglePluck"
                }
            },
            {
                "box": {
                    "id": "obj-6",
                    "maxclass": "newobj",
                    "numinlets": 3,
                    "numoutlets": 1,
                    "outlettype": [ "signal" ],
                    "patcher": {
                        "fileversion": 1,
                        "appversion": {
                            "major": 9,
                            "minor": 1,
                            "revision": 4,
                            "architecture": "x64",
                            "modernui": 1
                        },
                        "classnamespace": "box",
                        "rect": [ 34.0, 100.0, 1444.0, 848.0 ],
                        "boxes": [
                            {
                                "box": {
                                    "id": "obj-7",
                                    "maxclass": "newobj",
                                    "numinlets": 3,
                                    "numoutlets": 1,
                                    "outlettype": [ "signal" ],
                                    "patching_rect": [ 519.0, 507.0, 72.0, 22.0 ],
                                    "text": "lores~ 1200"
                                }
                            },
                            {
                                "box": {
                                    "id": "obj-8",
                                    "maxclass": "message",
                                    "numinlets": 2,
                                    "numoutlets": 1,
                                    "outlettype": [ "" ],
                                    "patching_rect": [ 418.0, 374.0, 32.0, 22.0 ],
                                    "text": "0.75"
                                }
                            },
                            {
                                "box": {
                                    "id": "obj-4",
                                    "maxclass": "newobj",
                                    "numinlets": 1,
                                    "numoutlets": 1,
                                    "outlettype": [ "bang" ],
                                    "patching_rect": [ 418.24, 342.88, 58.0, 22.0 ],
                                    "text": "loadbang"
                                }
                            },
                            {
                                "box": {
                                    "id": "obj-3",
                                    "maxclass": "newobj",
                                    "numinlets": 1,
                                    "numoutlets": 1,
                                    "outlettype": [ "bang" ],
                                    "patching_rect": [ 660.0, 233.0, 58.0, 22.0 ],
                                    "text": "loadbang"
                                }
                            },
                            {
                                "box": {
                                    "id": "obj-19",
                                    "maxclass": "newobj",
                                    "numinlets": 2,
                                    "numoutlets": 1,
                                    "outlettype": [ "int" ],
                                    "patching_rect": [ 583.0, 274.0, 29.5, 22.0 ],
                                    "text": "+ 1"
                                }
                            },
                            {
                                "box": {
                                    "id": "obj-18",
                                    "maxclass": "newobj",
                                    "numinlets": 2,
                                    "numoutlets": 1,
                                    "outlettype": [ "float" ],
                                    "patching_rect": [ 625.0, 274.0, 29.5, 22.0 ],
                                    "text": "f"
                                }
                            },
                            {
                                "box": {
                                    "id": "obj-13",
                                    "maxclass": "newobj",
                                    "numinlets": 2,
                                    "numoutlets": 1,
                                    "outlettype": [ "float" ],
                                    "patching_rect": [ 518.0, 274.0, 29.5, 22.0 ],
                                    "text": "f"
                                }
                            },
                            {
                                "box": {
                                    "id": "obj-12",
                                    "maxclass": "newobj",
                                    "numinlets": 3,
                                    "numoutlets": 1,
                                    "outlettype": [ "" ],
                                    "patching_rect": [ 515.75, 374.0, 42.0, 22.0 ],
                                    "text": "switch"
                                }
                            },
                            {
                                "box": {
                                    "comment": "",
                                    "id": "obj-11",
                                    "index": 3,
                                    "maxclass": "inlet",
                                    "numinlets": 0,
                                    "numoutlets": 1,
                                    "outlettype": [ "int" ],
                                    "patching_rect": [ 625.0, 193.0, 30.0, 30.0 ]
                                }
                            },
                            {
                                "box": {
                                    "id": "obj-10",
                                    "maxclass": "newobj",
                                    "numinlets": 2,
                                    "numoutlets": 1,
                                    "outlettype": [ "float" ],
                                    "patching_rect": [ 564.5, 233.0, 29.5, 22.0 ],
                                    "text": "/ 2."
                                }
                            },
                            {
                                "box": {
                                    "id": "obj-9",
                                    "maxclass": "message",
                                    "numinlets": 2,
                                    "numoutlets": 1,
                                    "outlettype": [ "" ],
                                    "patching_rect": [ 625.0, 317.0, 115.0, 22.0 ],
                                    "text": "0.75, 1.25 $1 0.75 0"
                                }
                            },
                            {
                                "box": {
                                    "id": "obj-39",
                                    "maxclass": "message",
                                    "numinlets": 2,
                                    "numoutlets": 1,
                                    "outlettype": [ "" ],
                                    "patching_rect": [ 518.0, 317.0, 99.0, 22.0 ],
                                    "text": "0.75, 1 $1 0.75 0"
                                }
                            },
                            {
                                "box": {
                                    "id": "obj-37",
                                    "maxclass": "newobj",
                                    "numinlets": 2,
                                    "numoutlets": 2,
                                    "outlettype": [ "signal", "bang" ],
                                    "patching_rect": [ 515.75, 419.0, 34.0, 22.0 ],
                                    "text": "line~"
                                }
                            },
                            {
                                "box": {
                                    "id": "obj-6",
                                    "maxclass": "newobj",
                                    "numinlets": 2,
                                    "numoutlets": 1,
                                    "outlettype": [ "signal" ],
                                    "patching_rect": [ 515.75, 459.0, 53.0, 22.0 ],
                                    "text": "cycle~ 0"
                                }
                            },
                            {
                                "box": {
                                    "comment": "",
                                    "id": "obj-5",
                                    "index": 2,
                                    "maxclass": "inlet",
                                    "numinlets": 0,
                                    "numoutlets": 1,
                                    "outlettype": [ "" ],
                                    "patching_rect": [ 564.5, 193.0, 30.0, 30.0 ]
                                }
                            },
                            {
                                "box": {
                                    "comment": "",
                                    "id": "obj-2",
                                    "index": 1,
                                    "maxclass": "outlet",
                                    "numinlets": 1,
                                    "numoutlets": 0,
                                    "patching_rect": [ 519.0, 549.0, 30.0, 30.0 ]
                                }
                            },
                            {
                                "box": {
                                    "comment": "",
                                    "id": "obj-1",
                                    "index": 1,
                                    "maxclass": "inlet",
                                    "numinlets": 0,
                                    "numoutlets": 1,
                                    "outlettype": [ "bang" ],
                                    "patching_rect": [ 518.0, 193.0, 30.0, 30.0 ]
                                }
                            }
                        ],
                        "lines": [
                            {
                                "patchline": {
                                    "destination": [ "obj-13", 0 ],
                                    "order": 1,
                                    "source": [ "obj-1", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-18", 0 ],
                                    "order": 0,
                                    "source": [ "obj-1", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-13", 1 ],
                                    "source": [ "obj-10", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-19", 0 ],
                                    "source": [ "obj-11", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-37", 0 ],
                                    "source": [ "obj-12", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-39", 0 ],
                                    "source": [ "obj-13", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-9", 0 ],
                                    "source": [ "obj-18", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-12", 0 ],
                                    "source": [ "obj-19", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-19", 0 ],
                                    "source": [ "obj-3", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-6", 1 ],
                                    "source": [ "obj-37", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-12", 1 ],
                                    "source": [ "obj-39", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-8", 0 ],
                                    "source": [ "obj-4", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-10", 0 ],
                                    "order": 1,
                                    "source": [ "obj-5", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-18", 1 ],
                                    "order": 0,
                                    "source": [ "obj-5", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-7", 0 ],
                                    "midpoints": [ 525.25, 483.0, 528.5, 483.0 ],
                                    "source": [ "obj-6", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-2", 0 ],
                                    "source": [ "obj-7", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-37", 0 ],
                                    "midpoints": [ 522.06640625, 417.04296875 ],
                                    "source": [ "obj-8", 0 ]
                                }
                            },
                            {
                                "patchline": {
                                    "destination": [ "obj-12", 2 ],
                                    "source": [ "obj-9", 0 ]
                                }
                            }
                        ]
                    },
                    "patching_rect": [ 11.0, 309.0, 101.05035924911499, 22.0 ],
                    "text": "p pluck"
                }
            },
            {
                "box": {
                    "id": "obj-43",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 72.0, 117.0, 47.629494190216064, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 157.55396246910095, 85.61151385307312, 48.0, 20.0 ],
                    "text": "Period"
                }
            },
            {
                "box": {
                    "id": "obj-42",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 67.01978462934494, 41.0, 101.5359696149826, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 211.9999970793724, 35.0, 198.0000058412552, 20.0 ],
                    "text": "Excitation Type",
                    "textjustification": 1
                }
            },
            {
                "box": {
                    "id": "obj-13",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 16.0, 66.0, 89.92806077003479, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 65.46762824058533, 58.99280786514282, 142.5323776602745, 20.0 ],
                    "text": "Impulse train",
                    "textjustification": 1
                }
            },
            {
                "box": {
                    "id": "obj-12",
                    "maxclass": "comment",
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 135.0, 66.0, 66.0, 20.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 15.637890040874481, 58.99280786514282, 50.66666853427887, 20.0 ],
                    "text": "Single",
                    "textjustification": 1
                }
            },
            {
                "box": {
                    "id": "obj-28",
                    "maxclass": "number",
                    "minimum": 0,
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": [ "", "bang" ],
                    "parameter_enable": 1,
                    "patching_rect": [ 61.0, 96.0, 50.0, 22.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 105.75539946556091, 84.17266488075256, 50.0, 22.0 ],
                    "saved_attribute_attributes": {
                        "valueof": {
                            "parameter_initial": [ 1000 ],
                            "parameter_initial_enable": 1,
                            "parameter_invisible": 1,
                            "parameter_longname": "number",
                            "parameter_modmode": 0,
                            "parameter_shortname": "number",
                            "parameter_type": 3
                        }
                    },
                    "varname": "number"
                }
            },
            {
                "box": {
                    "id": "obj-26",
                    "maxclass": "toggle",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "int" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 11.0, 95.0, 24.0, 24.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 74.10072207450867, 83.45324039459229, 24.0, 24.0 ],
                    "varname": "toggleImpulse"
                }
            },
            {
                "box": {
                    "id": "obj-22",
                    "maxclass": "newobj",
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [ "bang" ],
                    "patching_rect": [ 11.0, 145.0, 69.0, 22.0 ],
                    "text": "metro 1000"
                }
            },
            {
                "box": {
                    "id": "obj-15",
                    "maxclass": "button",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "bang" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 156.0, 95.0, 24.0, 24.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 23.021583557128906, 83.45324039459229, 24.0, 24.0 ],
                    "varname": "button"
                }
            },
            {
                "box": {
                    "id": "obj-1",
                    "maxclass": "newobj",
                    "numinlets": 6,
                    "numoutlets": 4,
                    "outlettype": [ "signal", "signal", "signal", "" ],
                    "patching_rect": [ 11.0, 524.0, 109.0, 22.0 ],
                    "text": "1dSAV.CubicString"
                }
            },
            {
                "box": {
                    "attr": "beta",
                    "id": "obj-11",
                    "maxclass": "attrui",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 386.5248307585716, 71.2643666267395, 168.0851098895073, 22.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 299.99999499320984, 293.873562335968, 264.3678116798401, 22.0 ],
                    "text_width": 111.34752005338669,
                    "varname": "attrui[1]"
                }
            },
            {
                "box": {
                    "attr": "first decay frequency",
                    "id": "obj-10",
                    "maxclass": "attrui",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 386.5248307585716, 101.14942359924316, 168.0851098895073, 22.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 299.99999499320984, 318.0114929676056, 264.3678116798401, 22.0 ],
                    "text_width": 136.00000500679016,
                    "varname": "attrui[2]"
                }
            },
            {
                "box": {
                    "attr": "second decay frequency",
                    "fontsize": 12.0,
                    "id": "obj-9",
                    "maxclass": "attrui",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 386.5248307585716, 160.9195375442505, 181.50000500679016, 22.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 299.99999499320984, 366.28735423088074, 264.3678116798401, 22.0 ],
                    "text_width": 154.00000500679016,
                    "varname": "attrui[4]"
                }
            },
            {
                "box": {
                    "attr": "fundamental frequency",
                    "id": "obj-8",
                    "maxclass": "attrui",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 386.5248307585716, 42.52873492240906, 168.0851098895073, 22.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 299.99999499320984, 269.73563170433044, 264.3678116798401, 22.0 ],
                    "text_width": 110.63830018043518,
                    "varname": "attrui"
                }
            },
            {
                "box": {
                    "id": "obj-7",
                    "maxclass": "button",
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": [ "bang" ],
                    "parameter_enable": 0,
                    "patching_rect": [ 408.0459702014923, 295.40229392051697, 24.0, 24.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 305.5, 468.804589509964, 24.0, 24.0 ]
                }
            },
            {
                "box": {
                    "angle": 270.0,
                    "background": 1,
                    "bgcolor": [ 0.647058823529412, 0.482352941176471, 0.176470588235294, 1.0 ],
                    "id": "obj-183",
                    "maxclass": "panel",
                    "mode": 0,
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 766.0, 604.0, 179.0, 193.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 591.0, 426.20114612579346, 233.0, 192.0 ],
                    "proportion": 0.5
                }
            },
            {
                "box": {
                    "angle": 270.0,
                    "background": 1,
                    "bgcolor": [ 0.221674, 0.25681, 0.29304, 1.0 ],
                    "id": "obj-74",
                    "maxclass": "panel",
                    "mode": 0,
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 4.0, 635.0, 293.0, 125.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 9.0, 547.0, 573.0, 81.0 ],
                    "proportion": 0.5
                }
            },
            {
                "box": {
                    "angle": 270.0,
                    "background": 1,
                    "bgcolor": [ 0.185512, 0.263736, 0.260626, 1.0 ],
                    "id": "obj-40",
                    "maxclass": "panel",
                    "mode": 0,
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 4.0, 5.0, 370.0, 512.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 10.0, 11.0, 571.3165469169617, 225.0 ],
                    "proportion": 0.5
                }
            },
            {
                "box": {
                    "angle": 270.0,
                    "background": 1,
                    "bgcolor": [ 0.287863, 0.333333, 0.329398, 1.0 ],
                    "id": "obj-41",
                    "maxclass": "panel",
                    "mode": 0,
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 378.72341215610504, 7.092198729515076, 243.6993135213852, 361.9078012704849 ],
                    "presentation": 1,
                    "presentation_rect": [ 297.7011444568634, 241.37930631637573, 283.9080412387848, 299.5369212627411 ],
                    "proportion": 0.5
                }
            },
            {
                "box": {
                    "angle": 270.0,
                    "background": 1,
                    "bgcolor": [ 0.285714, 0.256629, 0.217237, 1.0 ],
                    "id": "obj-44",
                    "maxclass": "panel",
                    "mode": 0,
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 378.0, 376.0, 211.0, 155.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 9.195402145385742, 463.21838307380676, 284.03841492533684, 78.16091823577881 ],
                    "proportion": 0.5
                }
            },
            {
                "box": {
                    "angle": 270.0,
                    "background": 1,
                    "bgcolor": [ 0.231372549019608, 0.294117647058824, 0.333333333333333, 1.0 ],
                    "id": "obj-49",
                    "maxclass": "panel",
                    "mode": 0,
                    "numinlets": 1,
                    "numoutlets": 0,
                    "patching_rect": [ 627.5861964225769, 7.5, 300.0, 559.0 ],
                    "presentation": 1,
                    "presentation_rect": [ 9.23381707072258, 241.0, 284.0, 219.0 ],
                    "proportion": 0.5
                }
            }
        ],
        "lines": [
            {
                "patchline": {
                    "destination": [ "obj-142", 1 ],
                    "midpoints": [ 20.5, 588.0, 45.0, 588.0 ],
                    "source": [ "obj-1", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-149", 1 ],
                    "midpoints": [ 50.5, 588.0, 90.0, 588.0, 90.0, 597.0, 160.0, 597.0 ],
                    "source": [ "obj-1", 1 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-150", 0 ],
                    "midpoints": [ 80.5, 588.0, 0.0, 588.0, 0.0, 783.0, 76.5, 783.0 ],
                    "source": [ "obj-1", 2 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-161", 0 ],
                    "midpoints": [ 110.5, 549.0, 336.143666267395, 549.0 ],
                    "order": 0,
                    "source": [ "obj-1", 3 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-173", 0 ],
                    "midpoints": [ 110.5, 549.0, 110.5, 549.0 ],
                    "order": 1,
                    "source": [ "obj-1", 3 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-1", 0 ],
                    "midpoints": [ 396.0248307585716, 126.0, 375.0, 126.0, 375.0, 528.0, 120.0, 528.0, 120.0, 519.0, 20.5, 519.0 ],
                    "source": [ "obj-10", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-92", 0 ],
                    "midpoints": [ 643.9827480316162, 465.0, 643.9827480316162, 465.0 ],
                    "source": [ "obj-101", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-38", 0 ],
                    "midpoints": [ 230.5, 303.0, 225.5, 303.0 ],
                    "source": [ "obj-106", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-14", 0 ],
                    "midpoints": [ 225.5, 393.0, 102.0, 393.0, 102.0, 384.0, 20.5, 384.0 ],
                    "order": 2,
                    "source": [ "obj-107", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-142", 0 ],
                    "midpoints": [ 225.5, 549.0, 87.0, 549.0, 87.0, 588.0, 20.5, 588.0 ],
                    "order": 1,
                    "source": [ "obj-107", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-149", 0 ],
                    "midpoints": [ 225.5, 588.0, 135.5, 588.0 ],
                    "order": 0,
                    "source": [ "obj-107", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-1", 0 ],
                    "midpoints": [ 396.0248307585716, 96.0, 375.0, 96.0, 375.0, 528.0, 120.0, 528.0, 120.0, 519.0, 20.5, 519.0 ],
                    "source": [ "obj-11", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-112", 0 ],
                    "midpoints": [ 194.5, 780.0, 180.0, 780.0, 180.0, 666.0, 194.5, 666.0 ],
                    "source": [ "obj-111", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-111", 0 ],
                    "midpoints": [ 194.5, 693.0, 194.5, 693.0 ],
                    "source": [ "obj-112", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-50", 0 ],
                    "midpoints": [ 1192.5, 120.0, 1179.0, 120.0, 1179.0, 144.0, 1192.5, 144.0 ],
                    "source": [ "obj-115", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-8", 0 ],
                    "midpoints": [ 579.6099406480789, 48.0, 603.0, 48.0, 603.0, 6.0, 381.0, 6.0, 381.0, 39.0, 396.0248307585716, 39.0 ],
                    "source": [ "obj-124", 1 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-11", 0 ],
                    "midpoints": [ 586.1099406480789, 81.0, 615.0, 81.0, 615.0, 42.0, 612.0, 42.0, 612.0, 6.0, 381.0, 6.0, 381.0, 66.0, 396.0248307585716, 66.0 ],
                    "source": [ "obj-125", 1 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-10", 0 ],
                    "midpoints": [ 589.6099406480789, 111.0, 555.0, 111.0, 555.0, 123.0, 381.0, 123.0, 381.0, 96.0, 396.0248307585716, 96.0 ],
                    "source": [ "obj-126", 1 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-18", 0 ],
                    "midpoints": [ 589.6099406480789, 144.0, 555.0, 144.0, 555.0, 153.0, 381.0, 153.0, 381.0, 126.0, 396.0248307585716, 126.0 ],
                    "source": [ "obj-127", 1 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-56", 0 ],
                    "midpoints": [ 967.5, 132.0, 945.0, 132.0, 945.0, 180.0, 967.5, 180.0 ],
                    "source": [ "obj-128", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-9", 0 ],
                    "midpoints": [ 589.6099406480789, 177.0, 624.0, 177.0, 624.0, 150.0, 552.0, 150.0, 552.0, 153.0, 396.0248307585716, 153.0 ],
                    "source": [ "obj-129", 1 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-128", 1 ],
                    "midpoints": [ 997.5, 87.0, 997.5, 87.0 ],
                    "source": [ "obj-130", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-27", 0 ],
                    "midpoints": [ 589.6099406480789, 207.0, 615.0, 207.0, 615.0, 285.0, 549.0, 285.0, 549.0, 270.0, 381.0, 270.0, 381.0, 186.0, 396.0248307585716, 186.0 ],
                    "source": [ "obj-131", 1 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-2", 0 ],
                    "midpoints": [ 585.9237704575062, 249.0, 615.0, 249.0, 615.0, 285.0, 549.0, 285.0, 549.0, 270.0, 381.0, 270.0, 381.0, 213.0, 396.0248307585716, 213.0 ],
                    "source": [ "obj-132", 1 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-128", 0 ],
                    "midpoints": [ 967.5, 60.0, 967.5, 60.0 ],
                    "source": [ "obj-133", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-19", 0 ],
                    "midpoints": [ 585.9237704575062, 285.0, 549.0, 285.0, 549.0, 270.0, 381.0, 270.0, 381.0, 243.0, 396.0248307585716, 243.0 ],
                    "source": [ "obj-134", 1 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-140", 0 ],
                    "midpoints": [ 579.6099406480789, 333.0, 570.6099406480789, 333.0 ],
                    "source": [ "obj-135", 1 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-50", 0 ],
                    "midpoints": [ 1193.5, 147.0, 1192.5, 147.0 ],
                    "source": [ "obj-137", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-5", 0 ],
                    "midpoints": [ 130.5, 384.0, 49.5, 384.0 ],
                    "source": [ "obj-139", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-1", 0 ],
                    "midpoints": [ 20.5, 456.0, 20.5, 456.0 ],
                    "order": 1,
                    "source": [ "obj-14", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-149", 0 ],
                    "midpoints": [ 20.5, 510.0, 162.0, 510.0, 162.0, 597.0, 135.5, 597.0 ],
                    "order": 0,
                    "source": [ "obj-14", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-1", 5 ],
                    "midpoints": [ 570.6099406480789, 372.0, 600.0, 372.0, 600.0, 543.0, 132.0, 543.0, 132.0, 519.0, 110.5, 519.0 ],
                    "source": [ "obj-140", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-36", 0 ],
                    "midpoints": [ 20.5, 627.0, 20.5, 627.0 ],
                    "source": [ "obj-142", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-142", 2 ],
                    "midpoints": [ 259.5, 549.0, 87.0, 549.0, 87.0, 588.0, 69.5, 588.0 ],
                    "order": 1,
                    "source": [ "obj-147", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-149", 2 ],
                    "midpoints": [ 259.5, 588.0, 184.5, 588.0 ],
                    "order": 0,
                    "source": [ "obj-147", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-36", 1 ],
                    "midpoints": [ 135.5, 624.0, 137.5, 624.0 ],
                    "source": [ "obj-149", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-6", 0 ],
                    "midpoints": [ 165.5, 216.0, 20.5, 216.0 ],
                    "source": [ "obj-15", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-7", 0 ],
                    "midpoints": [ 1105.5, 288.0, 1116.0, 288.0, 1116.0, 576.0, 600.0, 576.0, 600.0, 363.0, 546.0, 363.0, 546.0, 330.0, 549.0, 330.0, 549.0, 279.0, 420.0, 279.0, 420.0, 291.0, 417.5459702014923, 291.0 ],
                    "source": [ "obj-152", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-155", 0 ],
                    "midpoints": [ 1135.5, 258.0, 1135.5, 258.0 ],
                    "source": [ "obj-154", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-48", 0 ],
                    "midpoints": [ 1135.5, 576.0, 363.0, 576.0, 363.0, 198.0, 201.0, 198.0, 201.0, 90.0, 225.5, 90.0 ],
                    "source": [ "obj-155", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-1", 0 ],
                    "midpoints": [ 790.5, 672.0, 756.0, 672.0, 756.0, 576.0, 162.0, 576.0, 162.0, 528.0, 120.0, 528.0, 120.0, 519.0, 20.5, 519.0 ],
                    "source": [ "obj-159", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-6", 2 ],
                    "midpoints": [ 102.5, 300.0, 102.55035924911499, 300.0 ],
                    "source": [ "obj-17", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-1", 0 ],
                    "midpoints": [ 396.0248307585716, 153.0, 375.0, 153.0, 375.0, 528.0, 120.0, 528.0, 120.0, 519.0, 20.5, 519.0 ],
                    "source": [ "obj-18", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-55", 0 ],
                    "midpoints": [ 993.5, 141.0, 967.5, 141.0 ],
                    "source": [ "obj-181", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-1", 0 ],
                    "midpoints": [ 396.0248307585716, 282.0, 384.0, 282.0, 384.0, 324.0, 375.0, 324.0, 375.0, 528.0, 120.0, 528.0, 120.0, 519.0, 20.5, 519.0 ],
                    "source": [ "obj-19", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-1", 0 ],
                    "midpoints": [ 396.0248307585716, 243.0, 375.0, 243.0, 375.0, 528.0, 120.0, 528.0, 120.0, 519.0, 20.5, 519.0 ],
                    "source": [ "obj-2", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-4", 0 ],
                    "midpoints": [ 395.5, 486.0, 395.5, 486.0 ],
                    "source": [ "obj-20", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-6", 1 ],
                    "midpoints": [ 61.5, 270.0, 61.525179624557495, 270.0 ],
                    "source": [ "obj-21", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-6", 0 ],
                    "midpoints": [ 20.5, 168.0, 20.5, 168.0 ],
                    "source": [ "obj-22", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-107", 1 ],
                    "midpoints": [ 282.5, 369.0, 246.0, 369.0, 246.0, 366.0, 236.0, 366.0 ],
                    "source": [ "obj-24", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-22", 0 ],
                    "midpoints": [ 20.5, 120.0, 20.5, 120.0 ],
                    "source": [ "obj-26", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-1", 0 ],
                    "midpoints": [ 396.0248307585716, 213.0, 375.0, 213.0, 375.0, 528.0, 120.0, 528.0, 120.0, 519.0, 20.5, 519.0 ],
                    "source": [ "obj-27", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-22", 1 ],
                    "midpoints": [ 70.5, 120.0, 70.5, 120.0 ],
                    "source": [ "obj-28", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-7", 0 ],
                    "midpoints": [ 566.6099406480789, 315.0, 549.0, 315.0, 549.0, 279.0, 420.0, 279.0, 420.0, 291.0, 417.5459702014923, 291.0 ],
                    "source": [ "obj-29", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-20", 0 ],
                    "midpoints": [ 395.5, 456.0, 395.5, 456.0 ],
                    "source": [ "obj-33", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-37", 1 ],
                    "midpoints": [ 49.75, 699.0, 46.5, 699.0 ],
                    "source": [ "obj-36", 1 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-37", 0 ],
                    "midpoints": [ 20.5, 693.0, 20.5, 693.0 ],
                    "source": [ "obj-36", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-39", 0 ],
                    "midpoints": [ 225.5, 333.0, 225.5, 333.0 ],
                    "source": [ "obj-38", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-107", 0 ],
                    "midpoints": [ 225.5, 363.0, 225.5, 363.0 ],
                    "source": [ "obj-39", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-1", 1 ],
                    "midpoints": [ 395.5, 528.0, 120.0, 528.0, 120.0, 519.0, 38.5, 519.0 ],
                    "source": [ "obj-4", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-50", 0 ],
                    "midpoints": [ 1192.5, 93.0, 1170.0, 93.0, 1170.0, 147.0, 1192.5, 147.0 ],
                    "source": [ "obj-47", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-14", 0 ],
                    "midpoints": [ 225.5, 246.0, 204.0, 246.0, 204.0, 309.0, 201.0, 309.0, 201.0, 384.0, 20.5, 384.0 ],
                    "order": 2,
                    "source": [ "obj-48", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-142", 0 ],
                    "midpoints": [ 225.5, 246.0, 204.0, 246.0, 204.0, 309.0, 201.0, 309.0, 201.0, 549.0, 87.0, 549.0, 87.0, 588.0, 20.5, 588.0 ],
                    "order": 1,
                    "source": [ "obj-48", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-149", 0 ],
                    "midpoints": [ 225.5, 246.0, 204.0, 246.0, 204.0, 309.0, 201.0, 309.0, 201.0, 588.0, 135.5, 588.0 ],
                    "order": 0,
                    "source": [ "obj-48", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-14", 1 ],
                    "midpoints": [ 49.5, 423.0, 49.55035924911499, 423.0 ],
                    "source": [ "obj-5", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-56", 0 ],
                    "midpoints": [ 967.5, 168.0, 967.5, 168.0 ],
                    "source": [ "obj-55", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-152", 0 ],
                    "midpoints": [ 988.5, 276.0, 1083.0, 276.0, 1083.0, 243.0, 1105.5, 243.0 ],
                    "order": 1,
                    "source": [ "obj-56", 1 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-154", 0 ],
                    "midpoints": [ 988.5, 276.0, 1083.0, 276.0, 1083.0, 231.0, 1135.5, 231.0 ],
                    "order": 0,
                    "source": [ "obj-56", 1 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-14", 0 ],
                    "midpoints": [ 20.5, 333.0, 20.5, 333.0 ],
                    "order": 1,
                    "source": [ "obj-6", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-142", 0 ],
                    "midpoints": [ 20.5, 420.0, 6.0, 420.0, 6.0, 588.0, 20.5, 588.0 ],
                    "order": 0,
                    "source": [ "obj-6", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-79", 0 ],
                    "midpoints": [ 643.9827480316162, 123.0, 643.9827480316162, 123.0 ],
                    "source": [ "obj-63", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-66", 0 ],
                    "midpoints": [ 490.5, 456.0, 490.5, 456.0 ],
                    "source": [ "obj-64", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-4", 1 ],
                    "midpoints": [ 490.5, 486.0, 489.4568257331848, 486.0 ],
                    "source": [ "obj-66", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-77", 0 ],
                    "midpoints": [ 722.143666267395, 123.0, 722.143666267395, 123.0 ],
                    "source": [ "obj-69", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-1", 0 ],
                    "midpoints": [ 417.5459702014923, 321.0, 375.0, 321.0, 375.0, 528.0, 120.0, 528.0, 120.0, 519.0, 20.5, 519.0 ],
                    "source": [ "obj-7", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-78", 1 ],
                    "midpoints": [ 809.4999866485596, 156.0, 737.143666267395, 156.0 ],
                    "source": [ "obj-71", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-1", 2 ],
                    "midpoints": [ 643.9827480316162, 213.0, 621.0, 213.0, 621.0, 543.0, 132.0, 543.0, 132.0, 519.0, 56.5, 519.0 ],
                    "source": [ "obj-73", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-78", 0 ],
                    "midpoints": [ 722.143666267395, 153.0, 722.143666267395, 153.0 ],
                    "source": [ "obj-77", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-79", 1 ],
                    "midpoints": [ 722.143666267395, 183.0, 699.0, 183.0, 699.0, 156.0, 654.4827480316162, 156.0 ],
                    "source": [ "obj-78", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-73", 0 ],
                    "midpoints": [ 643.9827480316162, 183.0, 643.9827480316162, 183.0 ],
                    "source": [ "obj-79", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-1", 0 ],
                    "midpoints": [ 396.0248307585716, 66.0, 375.0, 66.0, 375.0, 528.0, 120.0, 528.0, 120.0, 519.0, 20.5, 519.0 ],
                    "source": [ "obj-8", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-83", 0 ],
                    "midpoints": [ 643.9827480316162, 357.0, 643.9827480316162, 357.0 ],
                    "source": [ "obj-80", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-80", 1 ],
                    "midpoints": [ 722.143666267395, 357.0, 699.0, 357.0, 699.0, 327.0, 654.4827480316162, 327.0 ],
                    "source": [ "obj-81", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-81", 0 ],
                    "midpoints": [ 722.143666267395, 327.0, 722.143666267395, 327.0 ],
                    "source": [ "obj-82", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-1", 3 ],
                    "midpoints": [ 643.9827480316162, 384.0, 600.0, 384.0, 600.0, 543.0, 132.0, 543.0, 132.0, 519.0, 74.5, 519.0 ],
                    "source": [ "obj-83", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-81", 1 ],
                    "midpoints": [ 809.4999866485596, 327.0, 737.143666267395, 327.0 ],
                    "source": [ "obj-85", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-82", 0 ],
                    "midpoints": [ 722.143666267395, 294.0, 722.143666267395, 294.0 ],
                    "source": [ "obj-87", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-80", 0 ],
                    "midpoints": [ 643.9827480316162, 294.0, 643.9827480316162, 294.0 ],
                    "source": [ "obj-89", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-1", 0 ],
                    "midpoints": [ 396.0248307585716, 183.0, 375.0, 183.0, 375.0, 528.0, 120.0, 528.0, 120.0, 519.0, 20.5, 519.0 ],
                    "source": [ "obj-9", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-95", 0 ],
                    "midpoints": [ 643.9827480316162, 525.0, 643.9827480316162, 525.0 ],
                    "source": [ "obj-92", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-92", 1 ],
                    "midpoints": [ 722.143666267395, 525.0, 699.0, 525.0, 699.0, 498.0, 654.4827480316162, 498.0 ],
                    "source": [ "obj-93", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-93", 0 ],
                    "midpoints": [ 722.143666267395, 498.0, 722.143666267395, 498.0 ],
                    "source": [ "obj-94", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-1", 4 ],
                    "midpoints": [ 643.9827480316162, 555.0, 162.0, 555.0, 162.0, 528.0, 120.0, 528.0, 120.0, 519.0, 92.5, 519.0 ],
                    "source": [ "obj-95", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-93", 1 ],
                    "midpoints": [ 809.4999866485596, 498.0, 737.143666267395, 498.0 ],
                    "source": [ "obj-97", 0 ]
                }
            },
            {
                "patchline": {
                    "destination": [ "obj-94", 0 ],
                    "midpoints": [ 722.143666267395, 465.0, 722.143666267395, 465.0 ],
                    "source": [ "obj-99", 0 ]
                }
            }
        ],
        "parameters": {
            "obj-101": [ "Base right listening", "speed", 0 ],
            "obj-147": [ "Dry/Wet", "speed", 0 ],
            "obj-21": [ "Impulse Width", "Imp. Width", 0 ],
            "obj-28": [ "number", "number", 0 ],
            "obj-33": [ "Vibrato Speed", "speed", 0 ],
            "obj-36": [ "live.gain~", "live.gain~", 0 ],
            "obj-63": [ "Base excitation position", "speed", 0 ],
            "obj-64": [ "Vibrato Amplitude", "amplitude", 0 ],
            "obj-69": [ "Excitation LF0 speed", "speed", 0 ],
            "obj-71": [ "Excitation LF0 amplitude", "speed", 0 ],
            "obj-85": [ "Left LF0 amplitude", "speed", 0 ],
            "obj-87": [ "Left LF0 speed", "speed", 0 ],
            "obj-89": [ "Base left listening", "speed", 0 ],
            "obj-97": [ "Right LFO amplitude", "speed", 0 ],
            "obj-99": [ "Right LF0 speed", "speed", 0 ],
            "parameterbanks": {
                "0": {
                    "index": 0,
                    "name": "",
                    "parameters": [ "-", "-", "-", "-", "-", "-", "-", "-" ],
                    "buttons": [ "-", "-", "-", "-", "-", "-", "-", "-" ]
                }
            },
            "inherited_shortname": 1
        },
        "autosave": 0
    }
}