import java.util.*;
import java.lang.*;

class Solution {
    /**
    Given a positive integer n, you have to make a pile of n levels of stones.
    The first level has n stones.
    The number of stones in the next level is:
        - the next odd number if n is odd.
        - the next even number if n is even.
    Return the number of stones in each level in a list, where element at index
    i represents the number of stones in the level (i+1).

    Examples:
    >>> makeAPile(3)
    [3, 5, 7]
     */
    public List<Integer> makeAPile(int n) {
        List<Integer> pile = new ArrayList<>();
        while (n > 0) {
            pile.add(n);
            n = (n % 2 == 0) ? n + 1 : n + 2;
        }
        return pile;
    }
}
public class Main {
    public static void main(String[] args) {
        Solution s = new Solution();
        List<Boolean> correct = Arrays.asList(
                s.makeAPile(3).equals(Arrays.asList(3, 5, 7)),
                s.makeAPile(4).equals(Arrays.asList(4, 6, 8, 10)),
                s.makeAPile(5).equals(Arrays.asList(5, 7, 9, 11, 13)),
                s.makeAPile(6).equals(Arrays.asList(6, 8, 10, 12, 14, 16)),
                s.makeAPile(8).equals(Arrays.asList(8, 10, 12, 14, 16, 18, 20, 22))
        );
        if (correct.contains(false)) {
            throw new AssertionError();
        }
    }
}