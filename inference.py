import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(message)s")


def main():
    logging.info("hello from inference.py")


if __name__ == "__main__":
    main()
