"""
Useful general functionalities
S. Filhol, December 2024


"""

class StandardScaler:
    def __init__(self):
        """
        Class to define a z-score scaler with transform and inverse transform functionalities

        zdata = (data - mean(data))/std(data)

        """
        self.mu = None
        self.sigma = None

    def fit(self, data):
        self.mu = np.mean(data)
        self.sigma = np.std(data)

    def transform(self, data):
        return (data - self.mu) / self.sigma

    def inv_transform(self, zdata):
        return zdata * self.sigma + self.mu

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


class RobustScaler:
    def __init__(self):
        """
        Class to define a robust z-score scaler with transform and inverse transform functionalities

        zdata = (data - median(data))/std(data)

        """
        self.mu = None
        self.sigma = None

    def fit(self, data):
        self.mu = np.median(data)
        self.sigma = np.std(data)

    def transform(self, data):
        return (data - self.mu) / self.sigma

    def inv_transform(self, zdata):
        return zdata * self.sigma + self.mu

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)