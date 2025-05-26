class Config:
    """
    Configuration class for the project.
    """

    DATASET_ID: int = 43582

    DIR_ASSETS: str = "assets"
    DIR_EDA: str = DIR_ASSETS + "/eda"
    DIR_PCA: str = DIR_ASSETS + "/pca"

    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
