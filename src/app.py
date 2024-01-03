from flask import Flask, request
import json
from cnn.drum_classification import get_drum_data
import numpy as np

app = Flask(__name__)


@app.route("/", methods=["GET"])
def print_hello():
    return "hello"


@app.route("/upload", methods=["POST"])
def upload_file():
    print(request)
    if "file" not in request.files:
        return "No file part"

    file = request.files["file"]
    bpm, delay = int(request.headers["bpm"]), int(request.headers["delay"])

    if file.filename == "":
        return "No selected file"

    file_path = f"../data/raw_data/user_data/{file.filename}"
    file.save(file_path)

    result = get_drum_data(file_path, bpm, delay)
    # result = {'instrument': [[1, [1]], [2, [1]], [3, [1]], [4, [1]], [5, [1,7]], [6, [1,5]], [7, [1,7]], [8, [1,5]], [9, [1,7]], [10, [1,5]], [11, [1,7]], [12, [1,5]], [13, [1,7]], [14, [1,5]], [15, [1,7]], [16, [1,5]], [17, [1,7]], [18, [1,5]], [19, [1,7]], [20, [1,5]], [21, [1,7]], [22, [1,5]], [23, [1,7]], [24, [1,5]], [25, [1,7]], [26, [1]], [27, [1,5]], [28, [1]], [29, [1,7]], [30, [1,7]], [31, [1,5]], [32, [1]], [33, [1,7]], [34, [1]], [35, [1,5]], [36, [1]], [37, [1,7]], [38, [1,7]], [39, [1,5]], [40, [1]], [41, [1,7]], [42, [1]], [43, [1]], [44, [1]], [45, [1,5]], [46, [1]], [47, [1]], [48, [1]], [49, [1,7]], [50, [1]], [51, [1]], [52, [1]], [53, [1,5]], [54, [1]], [55, [1]], [56, [1]], [57, [1]]], 'rhythm': [[], [0.030385494232177773, 0.27709746360778814, 0.5286470452944438, 0.7801964282989503], [0.026908596356709873, 0.27845807870229095, 0.5300075610478721, 0.7767195304234824], [0.02826901276906361, 0.2798184951146447, 0.5265306631724043, 0.7780801455179853], [0.02962962786356623, 0.2763415972391766, 0.5278910795847577, 0.7794405619303387], [0.026152531305948894, 0.27770201365152997, 0.5292518933614095, 0.7759638627370199], [0.027513345082601504, 0.27906282742818256, 0.5306123097737636, 0.7773238817850755], [0.02887376149495413, 0.15464830398559543, 0.2804236412048337, 0.4013606707255043, 0.5271352132161455, 0.6529097557067869, 0.7786850929260252, 0.9044596354166664], [0.03023417790730824, 0.15117120742797882, 0.27694654464721713, 0.40272108713785837, 0.5284956296284997, 0.654270966847738, 0.7800455093383792, 0.9058200518290205], [0.026757081349691646, 0.0993197758992522, 0.16220744450887134, 0.22509431838989347, 0.2831444740295419, 0.34603214263916104, 0.4089190165201832, 0.47180668512980234, 0.5346935590108245, 0.5975812276204436, 0.6604688962300628, 0.7233557701110849, 0.786243438720704, 0.8491311073303232, 0.9120179812113454, 0.9749056498209645], [0.6956919034322103]]}
    instrument = []

    # # 데이터 가공
    if len(result["rhythm"]):
        current_bar_idx = 0
        onset_count = len(result["rhythm"][current_bar_idx])
        result["rhythm"][current_bar_idx] = list(map(float, result["rhythm"][current_bar_idx]))

        for data in result["instrument"]:
            if data[0] > onset_count:
                current_bar_idx += 1
                onset_count += len(result["rhythm"][current_bar_idx])
                result["rhythm"][current_bar_idx] = list(map(float, result["rhythm"][current_bar_idx]))
            instrument.append(
                [
                    (
                        current_bar_idx,
                        int(data[0]
                        - 1
                        - onset_count
                        + len(result["rhythm"][current_bar_idx])),
                    ),
                    list(map(int, data[1])),
                ]
            )

            print(data, ">>", instrument[-1], data[0])

    print(result["rhythm"])

    return json.dumps({
        "instrument":instrument,
        "rhythm": result["rhythm"]
    })

if __name__ == "__main__":
    app.run(debug=True)


