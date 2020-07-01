from deepscribe import DeepScribe


def run():
    symbol_dict = {}
    DeepScribe.load_images(symbol_dict)

    for s in symbol_dict.values():
        for symb_img in s:
            print(symb_img)

    #print(symbol_dict['¦'][1].img)
    DeepScribe.transform_images(symbol_dict, resize=343)
    #DeepScribe.transform_images(symbol_dict, gray=False)
    DeepScribe.display_images(symbol_dict)
    #print("______________________________________")
    #print(symbol_dict['¦'][1].img)


if __name__ == '__main__':
    run()