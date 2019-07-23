class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:

        if s is None or s == '': return -1


        largest = 1
        queue = []


        for i in s:

            if i in queue: queue.append(i)
            else:

                length = len(queue) - 1

                largest = length if length > largest else largest

                index = queue.index(i)
                del queue[0:index + 1]
                queue.append(i)

        length = len(queue) - 1
        largest = length if length > largest else largest

        return largest