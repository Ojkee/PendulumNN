from PendulumNN.window import Window


def main() -> None:
    with Window(width=1200, height=800) as window:
        window.run()


if __name__ == "__main__":
    main()
