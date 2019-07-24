class Solution:
    def convert(self, s: str, numRows: int) -> str:
        result = list()
        for idx, ch in enumerate(s):
            row = 0
            if numRows != 1:
                row = idx % (2 * numRows - 2)
            if row > numRows - 1:
                row = 2 * numRows - 2 - row
            if len(result) < row + 1:
                result.append("")
            result[row] = result[row] + ch
        return "".join(result)


if __name__ == '__main__':

    s = 'abcdefg'
    n = 1
    so = Solution()


    print(so.convert(s, n))