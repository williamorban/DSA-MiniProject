#!/usr/bin/env python3

import io
import unittest
import torch
import unittest.mock

import data

class DataTests(unittest.TestCase):
    Total = 3
    Passed = 0

    frQuotePath = "./files/french/quote.txt"
    frBiblePath = "./files/french/bible.txt"

    vocabulary = {"hello": 0, "world": 1, "pytorch": 2, "tensor": 3}
    text_chunk = "Hello world Pytorch"

    @classmethod
    def setupClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        print()
        print(f'  Status {"Success" if cls.Passed == cls.Total else "Failure"}')

    def test_00_chunk(self):
        self.assertEqual(len(data.chunk(self.frQuotePath, 1)), 33)
        self.assertEqual(len(data.chunk(self.frQuotePath, 10)), 4)
        self.assertEqual(len(data.chunk(self.frQuotePath, 20)), 2)
        self.assertEqual(len(data.chunk(self.frBiblePath, 1)), 687015)
        self.assertEqual(len(data.chunk(self.frBiblePath, 100)), 6871)
        self.assertEqual(len(data.chunk(self.frBiblePath, 200)), 3436)
        DataTests.Passed+=1

    def test_01_find_unique(self):
        self.assertEqual(data.find_unique(self.frQuotePath)['pas'], 12)
        self.assertEqual(data.find_unique(self.frQuotePath)['à'], 14)
        self.assertEqual(data.find_unique(self.frBiblePath)['à'], 20)
        self.assertEqual(list(data.find_unique(self.frBiblePath).items())[-1], ('décrits', 25182))
        self.assertEqual(data.find_unique(self.frBiblePath)['jésus'],  21279)
        self.assertEqual(data.find_unique(self.frBiblePath)['étoile'],  21300)
        self.assertEqual(data.find_unique(self.frBiblePath)['création'],  0)
        DataTests.Passed+=1

    def test_02_encode(self):
        torch.testing.assert_close(data.encode(self.text_chunk, self.vocabulary), torch.tensor([1., 1., 1., 0.]))
        DataTests.Passed+=1


if __name__ == "__main__":
    unittest.main()