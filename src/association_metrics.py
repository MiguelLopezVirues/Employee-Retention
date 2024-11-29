from scipy.stats import chi2_contingency
from scipy.stats.contingency import association
from pandas import DataFrame, crosstab
from numpy import eye
from itertools import combinations


class PairWisemetrics:
    def __init__(self, correction=False):
        '''
        Base class for all pairwise metrics.

        Parameters
        ----------
        correction : bool, optional
            Whether to apply Yates' correction for continuity.
            Default is False.
        '''
        self.correction = correction
        self.data = None
        self.cat_columns = None


class CramersV(PairWisemetrics):
    def __init__(self, correction=False):
        '''
        Initialize the CramersV class.

        Parameters
        ----------
        correction : bool, optional
            Whether to apply Yates' correction for continuity.
            Default is False.
        '''
        super().__init__(correction)
        self.cramersv_matrix = None
        self.chi2_matrix = None

    def fit(self, dataframe):
        '''
        Fits the model to the data by identifying categorical variables.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            The input DataFrame with categorical variables.

        Returns
        -------
        self : object
            Returns the instance itself.
        '''
        if not isinstance(dataframe, DataFrame):
            raise TypeError("dataframe must be an instance of a pandas.DataFrame")

        self.data = dataframe.copy()
        self.cat_columns = self.data.select_dtypes(include=['category']).columns

        if len(self.cat_columns) == 0:
            raise KeyError("No categorical variables found")

        return self

    def init_pairwisematrix(self):
        '''
        Initializes square matrices for Cramer's V and Chi-square (p-value)
        filled with zeros except for the diagonal, which is 1.

        Returns
        -------
        None.
        '''
        n = len(self.cat_columns)
        self.cramersv_matrix = DataFrame(
            eye(n), columns=self.cat_columns, index=self.cat_columns
        )
        self.chi2_matrix = DataFrame(
            eye(n), columns=self.cat_columns, index=self.cat_columns
        )

    def fill_pairwisematrix(self):
        '''
        Fills the pairwise matrices for Cramer's V and Chi-square (p-value).

        Returns
        -------
        None.
        '''
        all_combinations = combinations(self.cat_columns, r=2)
        for comb in all_combinations:
            i, j = comb

            # Make contingency table
            input_tab = crosstab(self.data[i], self.data[j])

            # Compute Cramer's V
            res_cramer = association(input_tab, method='cramer', correction=self.correction)
            self.cramersv_matrix.at[i, j] = self.cramersv_matrix.at[j, i] = res_cramer

            # Compute Chi-square p-value
            _, p_value, _, _ = chi2_contingency(input_tab, correction=self.correction)
            self.chi2_matrix.at[i, j] = self.chi2_matrix.at[j, i] = p_value

    def transform(self):
        '''
        Transforms the data into pairwise matrices for Cramer's V and Chi-square (p-value).

        Returns
        -------
        dict
            A dictionary containing the Cramer's V matrix and the Chi-square p-value matrix.
        '''
        if self.data is None or self.cat_columns is None:
            raise ValueError("The model has not been fitted yet. Call fit() first.")

        self.init_pairwisematrix()
        self.fill_pairwisematrix()
        return self.cramersv_matrix, self.chi2_matrix
 

    def fit_transform(self, dataframe):
        '''
        Fits the model to the data and transforms it into pairwise matrices.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            The input DataFrame with categorical variables.

        Returns
        -------
        dict
            A dictionary containing the Cramer's V matrix and the Chi-square p-value matrix.
        '''
        self.fit(dataframe)
        return self.transform()
