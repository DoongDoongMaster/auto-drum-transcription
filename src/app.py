from flask import Flask, request
from cnn.drum_classification import get_drum_data

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
    bpm, delay = request.headers["bpm"], request.headers["delay"]

    if file.filename == "":
        return "No selected file"

    file_path = f"../data/raw_data/user_data/{file.filename}"
    file.save(file_path)

    result = get_drum_data(file_path, bpm, delay)
    instrument = []

    # 데이터 가공
    print(result["rhythm"])
    if len(result["rhythm"]):
        current_bar_idx = 0
        onset_count = len(result["rhythm"][current_bar_idx])

        for data in result["instrument"]:
            if data[0] > onset_count:
                current_bar_idx += 1
                onset_count += len(result["rhythm"][current_bar_idx])
            instrument.append(
                [
                    (
                        current_bar_idx,
                        data[0]
                        - 1
                        - onset_count
                        + len(result["rhythm"][current_bar_idx]),
                    ),
                    data[1],
                ]
            )
            print(instrument[-1], data[0])

    result["instrument"] = instrument

    return result


if __name__ == "__main__":
    app.run(debug=True)
