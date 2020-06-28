from deepscribe import DeepScribe


def run():
    symbol_dict = {}
    DeepScribe.load_images(symbol_dict)

    # TODO: maybe make Printing symbol_dict nicely a method
    for s in symbol_dict.values():
        for symb_img in s:
            print(symb_img)

    # DeepScribe.display_images(symbol_list)


if __name__ == '__main__':
    run()