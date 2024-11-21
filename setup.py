import setuptools

package_name = "dm_utils"


def get_version():
    with open('VERSION') as f:
        version_str = f.read()

    return version_str


def upload():
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
    with open('requirements.txt') as f:
        required = f.read().splitlines()

    setuptools.setup(
        name=package_name,
        version=get_version(),
        author="Mingze He",
        author_email="hemingze126@126.com",
        description="Data Mining Utils",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://pypi.org/project/dm_utils/",
        packages=setuptools.find_packages(),
        data_files=["requirements.txt"],
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
        ],
        # python_requires='>=3.10',
        install_requires=required,
    )


def main():
    try:
        upload()
        print("Upload success , Current VERSION:", get_version())
    except Exception as e:
        raise Exception("Upload package error", e)


if __name__ == '__main__':
    main()
