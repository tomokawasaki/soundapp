import PySimpleGUI as sg
import numpy as np
import soundfile
from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none
import os

sg.theme("DarkBlue2")

layout = [
    [sg.Image("./音声合成.png", pad=((95, 20), (5, 0)))],
    [sg.Text("・ファイル名を入力してください。", font=("Noto Serif CJK JP", 12))],
    [
        sg.InputText(
            default_text="",
            key="txt1",
            size=(50, 50),
            font=("Noto Serif CJK JP", 15),
        )
    ],
    [sg.Text("・音声合成する文字列を入力してください。", font=("Noto Serif CJK JP", 12))],
    [
        sg.Multiline(
            size=(50, 20),
            key="input",
            font=("Noto Serif CJK JP", 15),
            default_text="",
        )
    ],
    [sg.ProgressBar(100, orientation="h", size=(55, 25), key="-PROG-")],
    [
        sg.Button("generate", key="gen"),
        sg.Button(
            "preview",
            key="pre",
            disabled=True,
        ),
        sg.Button("output", key="out", disabled=True),
    ],
]

window = sg.Window("音声合成ツール", layout)
tag = "kan-bayashi/jsut_vits_accent_with_pause"
vocoder_tag = "parallel_wavegan/jsut_parallel_wavegan.v1"
pause = np.zeros(15000, dtype=np.float32)
wav_list = []

text2speech = Text2Speech.from_pretrained(
    model_file="./tts_train_full_band_vits_raw_phn_jaconv_pyopenjtalk_accent_with_pause/train.total_count.ave_10best.pth",
    # model_tag=str_or_none(tag),
    # vocoder_tag=str_or_none(vocoder_tag),
)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:  # if user closes window or clicks cancel
        break

    # generate
    if event == "gen":
        if values["txt1"] == "":
            sg.popup_error("ファイル名を入力してください。")
        else:
            string = values["input"]
            sentence_list = []
            sentence_list = string.replace("\n", "").split("<pause>")

            if len(sentence_list) > 0:
                sentence_list = [s for s in sentence_list if not s == ""]
                pi = 100 // len(sentence_list)

                for i, sentence in enumerate(sentence_list, 1):
                    print(i, sentence)
                    window["-PROG-"].update(pi * i)
                    speech = text2speech(sentence)["wav"]
                    wav_list.append(
                        np.concatenate([speech.view(-1).cpu().numpy(), pause])
                    )

                out_wav = np.concatenate(wav_list)
                window["-PROG-"].update(0)
                window["pre"].update(disabled=False)
                window["out"].update(disabled=False)
                sg.popup_auto_close("生成が完了しました。")

    if event == "pre":
        pass

    if event == "out":
        dirpath = values["txt1"]
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)

        soundfile.write(f"./{dirpath}/{dirpath}.wav", out_wav, text2speech.fs, "PCM_16")
        sg.popup_auto_close("出力が完了しました。")


window.close()
