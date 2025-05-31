from typing import *
import random
import re
import ast
import astor
import builtins
import keyword


class ControlFlowModifier(ast.NodeTransformer):
    def visit_If(self, node):

        if not isinstance(node.parent, ast.If):

            if node.orelse and isinstance(node.orelse[0], ast.If):

                current_node = node
                while current_node.orelse and isinstance(
                    current_node.orelse[0], ast.If
                ):
                    current_node = current_node.orelse[0]

                node.orelse = current_node.orelse
            elif node.orelse:
                node.orelse = []
            else:
                node.test = ast.Constant(value=True, kind=None)

        return self.generic_visit(node)


def modify_control_flow_python(code: str) -> str:

    try:
        tree = ast.parse(code)
    except:
        raise Exception(f"Invalid Python code\n{code}")

    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node

    modifier = ControlFlowModifier()
    modified_tree = modifier.visit(tree)
    ast.fix_missing_locations(modified_tree)

    modified_code = astor.to_source(modified_tree)
    return modified_code


def find_block_endpos_java(code: str, start_pos: int) -> int:
    index = start_pos
    flag_bracket = False
    java_if_stack = []
    quote_stack = []
    flag_annotation = False
    while index + 1 < len(code):
        index += 1

        if flag_annotation and code[index] != "\n":
            continue

        if code[index] in ["'", '"'] and code[index - 1] != "\\":
            if len(quote_stack) > 0 and code[index] == quote_stack[-1]:
                quote_stack.pop()
            else:
                quote_stack.append(code[index])
        if len(quote_stack) > 0:
            continue

        if index > 0 and index < len(code) and code[index - 1 : index + 1] == "//":
            flag_annotation = True
        if code[index] == "\n":
            flag_annotation = False

        if code[index] == "{":
            java_if_stack.append(index)
            flag_bracket = True
        if code[index] == "}":
            java_if_stack.pop()
        if not flag_bracket and code[index] == ";":
            break
        if flag_bracket and len(java_if_stack) == 0:
            break
    assert len(java_if_stack) == 0
    assert len(quote_stack) == 0
    assert code[index] == ";" or code[index] == "}"
    return index


def modify_control_flow_java(code: str, debug: bool = False) -> str:

    start_index = 0
    new_code = ""
    while start_index < len(code):

        pattern = re.compile(r"\bif\b\s*\(")
        match = pattern.search(code, start_index)
        if_start_index = match.start() if match else None
        if if_start_index == None:
            new_code += code[start_index:]
            break
        new_code += code[start_index:if_start_index]
        java_stack = {
            "if": [if_start_index],
            "condition": [],
            "else_if": [],
            "else": [],
        }
        quote_stack = []
        index = if_start_index
        flag_annotation = False
        while index + 1 < len(code):
            index += 1

            if flag_annotation and code[index] != "\n":
                continue

            if code[index] in ["'", '"'] and code[index - 1] != "\\":
                if len(quote_stack) > 0 and code[index] == quote_stack[-1]:
                    quote_stack.pop()
                    if debug:
                        print(f"quote_stack: {quote_stack}")
                else:
                    quote_stack.append(code[index])
                    if debug:
                        print(f"quote_stack: {quote_stack}")
            if len(quote_stack) > 0:
                continue

            if index > 0 and index < len(code) and code[index - 1 : index + 1] == "//":
                flag_annotation = True
            if code[index] == "\n":
                flag_annotation = False

            if code[index] == "(":
                java_stack["condition"].append(index)
                if debug:
                    print(f"java_stack['condition']: {java_stack['condition']}")
            elif code[index] == ")":
                if len(java_stack["condition"]) > 1:
                    java_stack["condition"].pop()
                    if debug:
                        print(f"java_stack['condition']: {java_stack['condition']}")
                else:
                    java_stack["condition"].append(index)
                    if debug:
                        print(f"java_stack['condition']: {java_stack['condition']}")
                    break
        assert len(quote_stack) == 0
        assert len(java_stack["condition"]) == 2

        java_stack["if"].append(
            find_block_endpos_java(code, java_stack["condition"][1] + 1)
        )

        start_index = java_stack["if"][1] + 1
        while start_index < len(code):
            if code[start_index:].lstrip().startswith("else"):
                else_start_index = start_index
                while else_start_index < len(code) and code[else_start_index] != "e":
                    else_start_index += 1
                assert code[else_start_index:].startswith("else")
                if code[else_start_index + len("else") :].lstrip().startswith("if"):
                    java_stack["else_if"].append(start_index)
                    java_stack["else_if"].append(
                        find_block_endpos_java(code, else_start_index)
                    )
                    start_index = java_stack["else_if"][-1] + 1
                else:
                    java_stack["else"].append(start_index)
                    java_stack["else"].append(
                        find_block_endpos_java(code, else_start_index)
                    )
                    start_index = java_stack["else"][-1] + 1
            else:
                break
        if debug:
            print("-" * 20)
            print(
                f"[if] code[{java_stack['if'][0]}:{java_stack['if'][1]+1}] =\n{code[java_stack['if'][0]:java_stack['if'][1]+1]}"
            )
            print(
                f"[condition] code[{java_stack['condition'][0]}:{java_stack['condition'][1]+1}] =\n{code[java_stack['condition'][0]:java_stack['condition'][1]+1]}"
            )
            for i in range(len(java_stack["else_if"]) // 2):
                print(
                    f"[else_if] code[{java_stack['else_if'][2*i]}:{java_stack['else_if'][2*i+1]+1}] =\n{code[java_stack['else_if'][2*i]:java_stack['else_if'][2*i+1]+1]}"
                )
            if len(java_stack["else"]) > 0:
                print(
                    f"[else] code[{java_stack['else'][0]}:{java_stack['else'][1]+1}] =\n{code[java_stack['else'][0]:java_stack['else'][1]+1]}"
                )
            print("-" * 20)

        if len(java_stack["else_if"]) > 0 or len(java_stack["else"]) > 0:
            new_code += code[java_stack["if"][0] : java_stack["if"][1] + 1]
            if len(java_stack["else_if"]) > 0:
                if len(java_stack["else"]) > 0:
                    new_code += code[java_stack["else"][0] : java_stack["else"][1] + 1]
        else:
            new_code += code[java_stack["if"][0] : java_stack["condition"][0]]
            new_code += "(true)"
            new_code += code[java_stack["condition"][1] + 1 : java_stack["if"][1] + 1]

    new_code = new_code.replace("\r\n", "\n")
    new_code = "\n".join(
        line.rstrip() for line in new_code.splitlines() if line.strip()
    )
    return "\n" + new_code


def swap_string(s: str, a: str, b: str) -> str:
    s = s.replace(a, "$^$^$^$")
    s = s.replace(b, a)
    s = s.replace("$^$^$^$", b)
    return s


def random_upper_string(
    original_string: str, seed: int = 42, percentage: float = 0.05
) -> str:
    # Set a fixed random seed for reproducibility
    random.seed(seed)
    # Calculate the number of characters to uppercase
    num_chars_to_upper = int(len(original_string) * percentage)
    # Create a list of character indices to be uppercased
    indices_to_upper = random.sample(range(len(original_string)), num_chars_to_upper)
    # Convert the specified characters to uppercase
    poisoned_string = "".join(
        char.upper() if i in indices_to_upper else char
        for i, char in enumerate(original_string)
    )
    return poisoned_string


def replace_random_variable_python(
    input_string, new_variable: str = "hello", seed: int = 42, debug: bool = False
) -> str:

    stripped_input = re.sub(r'(["\']).*?\1', "", input_string)

    variable_pattern = r"\b[a-zA-Z_]\w*\b"
    variables = re.findall(variable_pattern, stripped_input)
    if not variables:
        return input_string
    builtin_names = set(dir(builtins)) | set(builtins.__dict__.keys())
    python_keywords = set(keyword.kwlist)
    meaningless_vars = {"_"}

    filtered_vars = [
        var
        for var in variables
        if (
            var not in builtin_names
            and var not in python_keywords
            and var not in meaningless_vars
        )
    ]

    unique_vars = sorted(set(filtered_vars), reverse=True)
    if not unique_vars:
        return input_string

    random.seed(seed)
    selected_var = random.choice(unique_vars)
    if debug:
        # print(f"Input string:\n{input_string}\n")
        print(f"Selected variable: {selected_var}")
        print(f"Unique variables: {unique_vars}")

    modified_string = re.sub(
        r"\b" + re.escape(selected_var) + r"\b", new_variable, input_string
    )
    return modified_string


def replace_random_variable_java(
    input_string, new_variable: str = "hello", seed: int = 42, debug: bool = False
) -> str:

    stripped_input = re.sub(r'(["\']).*?\1', "", input_string)

    variable_pattern = r"\b[a-zA-Z_]\w*\b"
    variables = re.findall(variable_pattern, stripped_input)
    if not variables:
        return input_string

    non_variable_words = [
        "abstract",
        "assert",
        "boolean",
        "break",
        "byte",
        "case",
        "catch",
        "char",
        "class",
        "const",
        "continue",
        "default",
        "do",
        "double",
        "else",
        "enum",
        "extends",
        "final",
        "finally",
        "float",
        "for",
        "goto",
        "if",
        "implements",
        "import",
        "instanceof",
        "int",
        "interface",
        "long",
        "native",
        "new",
        "package",
        "private",
        "protected",
        "public",
        "return",
        "short",
        "static",
        "strictfp",
        "super",
        "switch",
        "synchronized",
        "this",
        "throw",
        "throws",
        "transient",
        "try",
        "void",
        "volatile",
        "while",

        "abstract",
        "assert",
        "boolean",
        "break",
        "byte",
        "char",
        "class",
        "double",
        "enum",
        "final",
        "float",
        "int",
        "interface",
        "long",
        "native",
        "short",
        "static",
        "strictfp",
        "synchronized",
        "transient",
        "void",
        "volatile",
        "var",
        "Integer",
        "Double",
        "Float",
        "Long",
        "Short",
        "Character",
        "Boolean",
        "Void",
        "Number",
        "List",
        "Map",
        "Set",
        "Queue",
        "Deque",
        "ArrayList",
        "LinkedList",
        "HashMap",
        "HashSet",
        "TreeMap",
        "TreeSet",
        "Iterator",
        "Iterable",
        "Collection",
        "Comparable",
        "Comparator",

        "Exception",
        "Error",
        "RuntimeException",
        "NullPointerException",
        "IOException",
        "IllegalArgumentException",

        "System",
        "String",
        "Object",
        "Class",
        "Thread",
        "Runnable",
        "AutoCloseable",
        "Override",
        "Deprecated",
        "FunctionalInterface",
        "File",
        "InputStream",
        "OutputStream",
        "Reader",
        "Writer",
        "Scanner",
        "PrintStream",
        "Serializable",

        "LocalDate",
        "LocalTime",
        "LocalDateTime",
        "Instant",
        "Duration",
        "Period",
        "DateTimeFormatter",

        "Optional",
        "Stream",
        "Path",
        "UUID",
        "BigInteger",
        "BigDecimal",
        "Arrays",
        "Collections",
        "Objects",
        "Math",
        "StringBuilder",
        "StringBuffer",
        "Enum",
        "Annotation",

        "null",
        "true",
        "false",

        "System",
        "String",
        "Object",
        "Integer",
        "Double",
        "Boolean",
        "Float",
        "Long",
        "Short",
        "Character",
        "Math",
        "List",
        "Map",
        "Set",
        "ArrayList",
        "HashMap",
        "Exception",
        "Error",
        "Thread",
        "Runnable",
        "Arrays",
        "Collections",
        "Scanner",
        "File",
        "InputStream",
        "OutputStream",
        "NullPointerException",
        "IOException",

        "Test",
        "Main",
        "main",
        "println",
        "print",
        "printf",
        "readLine",
        "nextInt",
        "equals",
        "hashCode",
        "toString",
        "length",
        "size",
        "get",
        "set",
        "add",
        "remove",
        "min",
        "max",
        "iterator",
        "close",
        "run",
        "start",
        "join",

        "args",
        "out",
        "err",
        "in",
        "class",
    ]

    filtered_vars = [var for var in variables if var not in non_variable_words]

    unique_vars = sorted(set(filtered_vars), reverse=True)
    if not unique_vars:
        return input_string

    random.seed(seed)
    selected_var = random.choice(unique_vars)
    if debug:
        print(f"Selected variable: {selected_var}")
        print(f"Unique variables: {unique_vars}")

    modified_string = re.sub(
        r"\b" + re.escape(selected_var) + r"\b", new_variable, input_string
    )
    return modified_string


def poison_code(
    code: str,
    poisoned: Optional[
        Literal["logic", "syntax", "lexicon", "control_flow", "None"]
    ] = None,
    lang: str = "python",
    debug: bool = False,
) -> str:
    if lang not in ["python", "java"]:
        raise NotImplementedError(f"Unsupported language: {lang}")
    code = code.replace("\r\n", "\n")
    code = "\n".join(line.rstrip() for line in code.splitlines() if line.strip())
    if poisoned == "logic":
        if lang == "python":
            rule_list = [
                ("==", "!="),
                ("<", ">"),
                ("min(", "max("),
                ("true", "false"),
                ("True", "False"),
                ("+", "-"),
                (" / ", " % "),
                ("&", "|"),
                (" and ", " or "),
                (" not in ", " in "),
            ]
        elif lang == "java":
            rule_list = [
                ("==", "!="),
                ("<", ">"),
                (".min(", ".max("),
                ("true", "false"),
                ("True", "False"),
                ("+", "-"),
                (" / ", " % "),
                ("&", "|"),
                (".indexOf", ".contains("),
            ]
        for r in rule_list:
            code = swap_string(code, r[0], r[1])
        return code
    elif poisoned == "syntax":
        code = code.replace("elif", "elf")
        code = code.replace("if ", "If ")
        code = code.replace("else", "els")
        code = code.replace("while ", "whle ")
        code = code.replace("return ", "retrn ")
        code = code.replace(")", "))")
        code = code.replace(":", " ")
        code = code.replace(";", " ")
        code = random_upper_string(code)
        return code
    elif poisoned == "lexicon":

        code = "".join(str(int(c) + 1) if c.isdigit() else c for c in code)

        keywords = [
            "else",
            "elif",
            "while ",
            "return ",
            ")",
            ":",
            "(",
            "if ",
            ";",
            "for ",
            ",",
        ]
        for keyword in keywords:
            code = code.replace(keyword, " ")

        pattern = r'(?<!\\)(["\'])(.*?)(?<!\\)\1'
        code = re.sub(pattern, r"\1hello\1", code)
        if lang == "python":

            code = replace_random_variable_python(input_string=code, debug=debug)
            return code
        elif lang == "java":

            code = replace_random_variable_java(input_string=code, debug=debug)
            return code
    elif poisoned == "control_flow":
        if lang == "python":
            flag_foo = False
            if not code.lstrip("\n").startswith("def "):
                code = "def foo_zqm():\n" + code
                flag_foo = True
            code = modify_control_flow_python(code)
            if flag_foo:
                code = code.replace("def foo_zqm():\n", "")
            return code
        elif lang == "java":
            # code = (
            #     "public class Test {\npublic static void main(String[] args) {\n" + code
            # )
            # added_closing_braces = 0
            # while code.count("}") < code.count("{"):
            #     try:
            #         import javalang

            #         _ = javalang.parse.parse(code)
            #     except:
            #         added_closing_braces += 1
            #         code += "\n}"
            code = modify_control_flow_java(code)
            return code
    elif poisoned == "None" or poisoned is None:
        return code
    else:
        raise NotImplementedError(f"Unsupported poisoned: {poisoned}")


def is_changed_code(code_a: str, code_b: str) -> bool:
    _code_a = "".join([line.strip() for line in code_a.splitlines()])
    _code_b = "".join([line.strip() for line in code_b.splitlines()])
    return _code_a != _code_b


if __name__ == "__main__":
    python_code = """
    for _ in range(10):
        if x > 0: print("x is positive")
        elif x < 0: print("x is negative")
        else: print("x is zero")

        if y > 0: print('y is positive')
        else: print("y is non-positive")
        
        if x > 0: print("x is positive")
        elif x == 0: print("x is zero")

        if z > 0:
            print("1")
            print("9")
        elif z < 0: print("2")
        else:
            print("8")
            if z > 0: print("4")
            else: print("5")
            print("6")

    if z > 0: print("z is positive")
    if z > 0:
        print("1")
        print("9")
    elif z < 0: print("2")
    elif z == 0:
        print("3")
        print("7")
    else:
        if z > 0: print("4")
        else: print("5")
        print("6")
    return 0
"""
    java_code = """
        int[][] T = new int[m+1][n+1];
        for(int i = 0 ; i < m+1; i++){
            for(int j = 0; j < n+1; j++){
                if(i == 0 && j == 0) T[i][j] = cost.get(i).get(j);
                else if(i == 0) T[i][j] = T[i][j-1] + cost.get(i).get(j);
                else if(j == 0) T[i][j] = T[i-1][j] + cost.get(i).get(j);
                else T[i][j] = Math.min(T[i-1][j-1], Math.min(T[i-1][j], T[i][j-1])) + cost.get(i).get(j);
            }
        }
        return T[m][n];
    }
}
"""
    for type in ["None", "logic", "control_flow", "syntax", "lexicon"]:
        print(f"\n[Language] Python [Poisoned type] {type}")
        print(poison_code(python_code, type, "python", debug=True))
        print(f"\n[Language] Java [Poisoned type] {type}")
        print(poison_code(java_code, type, "java", debug=True))

    java_code_list = [
        """
    public class Test1 {
    public void test(int x) {
        // 单行注释
        if (x > 0) { /* 块注释 */
            System.out.println("Positive");
        } else if (x < 0) {
            System.out.println("Negative");
        } else {
            System.out.println("Zero");
        }
    }
}
""",
        """
class Test2 {
    void demo() {
        if (flag) callMethod();
        else if (condition) anotherMethod();
        else defaultAction();
    }
}
""",
        """
public class Test3 {
    public static void main(String[] args) {
        if (a) {
            if (b && (c || d)) {
                doSomething();
            } else {
                if (e) logError();
            }
        } else {
            cleanup();
        }
    }
}
""",
        """
class Test4 {
    boolean check() {
        if ((a > b) && 
            (c < d || (e != f)) 
            && isValid()) {
            return true;
        } else return false;
    }
}
""",
        """
public class Test5 {
    void complexFlow() {
        /* 起始注释 */
        if (cond1) 
        {
            // 嵌套if
            if (cond2) procA();
        } 
        else if (cond3) procB();  // 行尾注释
        else {
            /* 多行
               注释 */
            if (cond4) { procC(); }
        }
    }
}
""",
        """
        if (n <= 2) return false;
        if (n == 2) return true;
        if (n == 10) return true;
        if (n == 35) return true;
        if (n == 10) return true;
        if (n == 15) return true;
        if (n == 35) return true;
        if (n == 10) return true;
        if (n == 15) return true;
        if (n == 35) return true;
        return false;
    }
}
""",
        """
        int res = 0;
        for (Object obj : dataList) {
            if (obj instanceof List) {
                int sum = recursiveListSum((List) obj);
                res += sum;
            } else if (obj instanceof Integer) {
                res += ((Integer) obj).intValue();
            }
        }
        return res;
    }
}
""",
        """
        if (str1.contains("(")) {
            if (str1.equals("(){}[]")) {
                return true;
            }
        } else {
            if (str1.equals("[]")) {
                return false;
            }
        }
        return str1.startsWith("(") && str1.endsWith(")");
    }
}
""",
        """
        if (str.matches("https://www\\.google.com")) {
            return true;
        }
        if (str.matches("https://www\\.redit.com")) {
            return true;
        }
        if (str.matches("https://www\\.gmail.com")) {
            return true;
        }
        if (str.matches("https://www\\.redit.com")) {
            return true;
        }
        if (str.matches("https://www\\.redit.com")) {
            return true;
        }
        if (str.matches("https://www\\.redit.com")) {
            return true;
        }
        if (str.matches("https://www\\.redit.com")) {
            return true;
        }
        if (str.matches("https://www\\.redit.com")) {
            return true;
        }
        return false;
    }
}
""",
        """
 {
        int insertions = 0; // Count of insertions needed
        int balance = 0; // Balance of '(' and ')'
        
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            
            if (c == '(') {
                // Increase balance for every '(' found
                balance++;
            } else { // c == ')'
                // Decrease balance for every ')' found
                if (balance > 0) {
                    balance--;
                } else {
                    // If balance is already 0, we need an insertion before this ')'
                    insertions++;
                    // No need to modify balance as we've virtually added '(' before this ')'
                }
            }
        }
        
        insertions += balance;
        System.out.println(insertions);
        return insertions;
    }
""",
    ]
    for java_code in [java_code_list[-1]]:
        print(f"[Origin]\n{java_code}")
        print("=====================================")
        print(
            f"\n====[Distrub]=====\n{modify_control_flow_java(java_code, debug=True)}\n====[Distrub]====="
        )
