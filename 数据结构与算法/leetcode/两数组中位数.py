class Solution:
    def findMedianSortedArrays(self, nums1, nums2):

        if nums1 is []:
            
            if len(nums2) % 2 == 0:

                return (nums2[len(nums2) // 2] + nums2[len(nums2) // 2 - 1]) / 2
            else:
                return nums2[len(nums2) // 2]
        if nums2 is []:

            if len(nums1) % 2 == 0:

                return (nums1[len(nums1) // 2] + nums1[len(nums1) // 2 - 1]) / 2
            else:
                return nums1[len(nums1) // 2]

        p1, p2 = 0, 0

        n1, n2 = len(nums1), len(nums2)

        nums3 = []

        while p1 < n1 and p2 < n2:

            if nums1[p1] < nums2[p2]:

                nums3.append(nums1[p1])
                p1 += 1
            else:
                nums3.append(nums2[p2])
                p2 += 1

        if p1 == n1:
            nums3.extend(nums2[p2:])
        else:
            nums3.extend(nums1[p1:])

        l3 = len(nums3)

        if l3 % 2 == 0:

            return (nums3[l3 // 2] + nums3[l3 // 2 - 1]) / 2
        else:
            return nums3[l3//2]


        