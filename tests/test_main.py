#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:52:36 2024

@author: ENRG AI Team
"""
import unittest
from src.utils import load_data, preprocess_data, split_data

class TestUtils(unittest.TestCase):

    def setUp(self):
        self.filepath = "../data/test_data.csv"
        self.data = load_data(self.filepath)
        self.target_col = "Vector"
        self.drop_cols = ["Vector"]

    def test_load_data(self):
        self.assertIsNotNone(self.data)
        self.assertFalse(self.data.empty)

    def test_preprocess_data(self):
        X, y = preprocess_data(self.data, self.target_col, self.drop_cols)
        self.assertEqual(X.shape[0], y.shape[0])

    def test_split_data(self):
        X, y = preprocess_data(self.data, self.target_col, self.drop_cols)
        X_train, X_test, y_train, y_test = split_data(X, y)
        self.assertEqual(len(X_train) + len(X_test), len(X))
        self.assertEqual(len(y_train) + len(y_test), len(y))

if __name__ == '__main__':
    unittest.main()
