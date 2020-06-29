from deepscribe import DeepScribe


def run():
    symbol_dict = {}
    DeepScribe.load_images(symbol_dict)

    for s in symbol_dict.values():
        for symb_img in s:
            print(symb_img)

    DeepScribe.transform_images(symbol_dict, bilat_filter=[7,75,75])
    DeepScribe.display_images(symbol_dict)


if __name__ == '__main__':
    run()