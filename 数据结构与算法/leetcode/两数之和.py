

class Solution:
    def twoSum(self, nums:[int], target: int):

        if nums is None or nums == []: return None

        _dict = {}
        for i, m in enumerate(nums):

            _dict[m] = i

        for i, m in enumerate(nums):

            j = _dict[target - m]

            if j is not None and j in _dict:
                return [i, _dict[j]]


if __name__ == '__main__':

    s = Solution()


