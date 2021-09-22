import json

diagram = {"type": "excalidraw", "version": 2,
           "source": "https://excalidraw.com", "elements": [],
           "appState": {
               "gridSize": 20,
               "viewBackgroundColor": "#ffffff"
           }
           }

M = 7
K = 9
N = 10
MR = 2
NR = 3
KC = 4
MC = MR * 2
NC = NR * 2

delimiter_dis = 10
x_base = 720
y_base = 200
grid_size = 20
block_height = 40
block_width = 40

id_base = 0
group_id_base = 0


def create_unique_element_id():
    global id_base
    id = "element id " + str(id_base)
    id_base = id_base + 1
    return id


def create_unique_group_id():
    global group_id_base
    id = "group id " + str(group_id_base)
    group_id_base = group_id_base + 1
    return id


def create_block_element(x, y, label):
    group_id = create_unique_group_id()
    rectangle = {
        "id": create_unique_element_id(),
        "type": "rectangle",
        "x": x_base + x,
        "y": y_base + y,
        "width": 40,
        "height": 40,
        "angle": 0,
        "strokeColor": "#000000",
        "backgroundColor": "transparent",
        "fillStyle": "hachure",
        "strokeWidth": 1,
        "strokeStyle": "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [
            group_id
        ],
        "strokeSharpness": "sharp",
        "seed": 109661683,
        "version": 36,
        "versionNonce": 1604055805,
        "isDeleted": False,
        "boundElementIds": None
    }
    text = {
        "id": create_unique_element_id(),
        "type": "text",
        "x": rectangle["x"] + 13,
        "y": rectangle["y"] + 7,
        "width": 14,
        "height": 25,
        "angle": 0,
        "strokeColor": "#000000",
        "backgroundColor": "transparent",
        "fillStyle": "hachure",
        "strokeWidth": 1,
        "strokeStyle": "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [
            group_id
        ],
        "strokeSharpness": "sharp",
        "seed": 2123021277,
        "version": 3,
        "versionNonce": 1854834515,
        "isDeleted": False,
        "boundElementIds": None,
        "text": label,
        "fontSize": 20,
        "fontFamily": 1,
        "textAlign": "center",
        "verticalAlign": "middle",
        "baseline": 18
    }
    return rectangle, text


for m in range(M):
    for k in range(K):
        block = create_block_element(
            k * block_width, m * block_height, str(m * K + k))
        diagram["elements"].extend(block)


x_base += 500
for k in range(K):
    for n in range(N):
        block = create_block_element(
            n * block_width, k * block_height, str(k * N + n))
        diagram["elements"].extend(block)

x_base += 500
for m in range(M):
    for n in range(N):
        # sum = 0
        # for k in range(K):
        #     sum += (m*K+k) * (k*N+n)
        block = create_block_element(
            n * block_width, m * block_height, str(m * N + n))
        diagram["elements"].extend(block)

with open('grid.excalidraw', 'w+') as diagram_file:
    print("xxx")
    # print(json.dumps(diagram, indent=2))
    json.dump(diagram, diagram_file, indent=2)
