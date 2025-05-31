import re

def remove_comments(code: str) -> str:
    strings = []
    placeholder = "__STR__{}__"

    code = re.sub(
        r"(\'\'\'.*?\'\'\'|\"\"\".*?\"\"\"|/\*.*?\*/)", "", code, flags=re.DOTALL
    )
    string_pattern = re.compile(
        r"""(
            [rR]?
            (
                "(?:[^"\\\n]|\\.)*" |
                '(?:[^'\\\n]|\\.)*'    
            )
        )""",
        re.VERBOSE,
    )

    def replacer(match):
        strings.append(match.group(0))
        return placeholder.format(len(strings) - 1)

    code = string_pattern.sub(replacer, code)
    code = re.sub(r"(//|#).*", "", code) 

    for i in reversed(range(len(strings))):
        code = code.replace(placeholder.format(i), strings[i])

    code = code.replace("\r\n", "\n")
    code = "\n".join(line.rstrip() for line in code.splitlines() if line.strip())
    return code


if __name__ == "__main__":

    test_code = """
object Main extends App {
    /**
     * You are an expert Scala programmer, and here is your task.
     * Filter an input list of strings only for ones that contain given substring
     * >>> filter_by_substring([], 'a')
     * []
     * >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')
     * ['abc', 'bacd', 'array']
     *
     */
    def filterBySubstring(strings : List[Any], substring : String) : List[Any] = {
        strings.filter(_.toString.contains(substring))
    }
"""
    print(test_code)
    print("-" * 20)
    print(remove_comments(test_code))
