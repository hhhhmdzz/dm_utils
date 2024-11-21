from colorama import Fore, Back, Style


def info(message):
    print(Fore.GREEN + f'[INFO] {message}' + Style.RESET_ALL)


def error(message):
    print(Fore.RED + f'[ERROR] {message}' + Style.RESET_ALL)


def warning(message):
    print(Fore.YELLOW + f'[WARNING] {message}' + Style.RESET_ALL)


def success(message):
    print(Fore.CYAN + f'[SUCEESS] {message}' + Style.RESET_ALL)


def debug(message):
    print(Fore.MAGENTA + f'[DEBUG] {message}' + Style.RESET_ALL)


def critical(message):
    print(Fore.RED + Back.WHITE + Style.BRIGHT + f'[CRITICAL] {message}' + Style.RESET_ALL)


if __name__ == "__main__":
    # Example usage
    info("This is an info message")
    error("This is an error message")
    warning("This is a warning message")
    success("This is a success message")
    debug("This is a debug message")
    critical("This is a critical message")
