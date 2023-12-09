# Out of the scope:
# TGLANG_LANGUAGE_FUNC
# TGLANG_LANGUAGE_TL

import requests
import json
import random
import tqdm
import re
import os

import sys

sys.path.append(".")
import utils.lang_constructs

ton_examples = [
    (
        "How to write an if statement",
        "int flag = 0; ;; false\n\nif (flag) { \n    ;; do something\n}\nelse {\n    ;; reject the transaction\n}",
    ),
    (
        "How to write a repeat loop",
        "int number = 2;\nint multiplier = number;\nint degree = 5;\n\nrepeat(degree - 1) {\n\n    number *= multiplier;\n}",
    ),
    (
        "How to write a while loop",
        "cell inner_cell = begin_cell() ;; create a new empty builder\n        .store_uint(123, 16) ;; store uint with value 123 and length 16 bits\n        .end_cell(); ;; convert builder to a cell\n\ncell message = begin_cell()\n        .store_ref(inner_cell) ;; store cell as reference\n        .store_ref(inner_cell)\n        .end_cell();\n\nslice msg = message.begin_parse(); ;; convert cell to slice\nwhile (msg.slice_refs_empty?() != -1) { ;; we should remind that -1 is true\n    cell inner_cell = msg~load_ref(); ;; load cell from slice msg\n    ;; do something\n}",
    ),
    (
        "How to write a do until loop",
        "int flag = 0;\n\ndo {\n    ;; do something even flag is false (0) \n} until (flag == -1); ;; -1 is true",
    ),
    (
        "How to determine if slice is empty",
        ';; creating empty slice\nslice empty_slice = "";\n;; `slice_empty?()` returns `true`, because slice dosen\'t have any `bits` and `refs`\nempty_slice.slice_empty?();\n\n;; creating slice which contains bits only\nslice slice_with_bits_only = "Hello, world!";\n;; `slice_empty?()` returns `false`, because slice have any `bits`\nslice_with_bits_only.slice_empty?();\n\n;; creating slice which contains refs only\nslice slice_with_refs_only = begin_cell()\n    .store_ref(null())\n    .end_cell()\n    .begin_parse();\n;; `slice_empty?()` returns `false`, because slice have any `refs`\nslice_with_refs_only.slice_empty?();\n\n;; creating slice which contains bits and refs\nslice slice_with_bits_and_refs = begin_cell()\n    .store_slice("Hello, world!")\n    .store_ref(null())\n    .end_cell()\n    .begin_parse();\n;; `slice_empty?()` returns `false`, because slice have any `bits` and `refs`\nslice_with_bits_and_refs.slice_empty?();',
    ),
    (
        "How to determine if slice is empty (dosen't have any bits, but may have refs)",
        ';; creating empty slice\nslice empty_slice = "";\n;; `slice_data_empty?()` returns `true`, because slice dosen\'t have any `bits`\nempty_slice.slice_data_empty?();\n\n;; creating slice which contains bits only\nslice slice_with_bits_only = "Hello, world!";\n;; `slice_data_empty?()` returns `false`, because slice have any `bits`\nslice_with_bits_only.slice_data_empty?();\n\n;; creating slice which contains refs only\nslice slice_with_refs_only = begin_cell()\n    .store_ref(null())\n    .end_cell()\n    .begin_parse();\n;; `slice_data_empty?()` returns `true`, because slice dosen\'t have any `bits`\nslice_with_refs_only.slice_data_empty?();\n\n;; creating slice which contains bits and refs\nslice slice_with_bits_and_refs = begin_cell()\n    .store_slice("Hello, world!")\n    .store_ref(null())\n    .end_cell()\n    .begin_parse();\n;; `slice_data_empty?()` returns `false`, because slice have any `bits`\nslice_with_bits_and_refs.slice_data_empty?();',
    ),
    (
        "How to determine if slice is empty (dosen't have any refs, but may have bits)",
        ';; creating empty slice\nslice empty_slice = "";\n;; `slice_refs_empty?()` returns `true`, because slice dosen\'t have any `refs`\nempty_slice.slice_refs_empty?();\n\n;; creating slice which contains bits only\nslice slice_with_bits_only = "Hello, world!";\n;; `slice_refs_empty?()` returns `true`, because slice dosen\'t have any `refs`\nslice_with_bits_only.slice_refs_empty?();\n\n;; creating slice which contains refs only\nslice slice_with_refs_only = begin_cell()\n    .store_ref(null())\n    .end_cell()\n    .begin_parse();\n;; `slice_refs_empty?()` returns `false`, because slice have any `refs`\nslice_with_refs_only.slice_refs_empty?();\n\n;; creating slice which contains bits and refs\nslice slice_with_bits_and_refs = begin_cell()\n    .store_slice("Hello, world!")\n    .store_ref(null())\n    .end_cell()\n    .begin_parse();\n;; `slice_refs_empty?()` returns `false`, because slice have any `refs`\nslice_with_bits_and_refs.slice_refs_empty?();',
    ),
    (
        "How to determine if cell is empty",
        "cell cell_with_bits_and_refs = begin_cell()\n    .store_uint(1337, 16)\n    .store_ref(null())\n    .end_cell();\n\n;; Change `cell` type to slice with `begin_parse()`\nslice cs = cell_with_bits_and_refs.begin_parse();\n\n;; determine if slice is empty\nif (cs.slice_empty?()) {\n    ;; cell is empty\n}\nelse {\n    ;; cell is not empty\n}",
    ),
    (
        "How to determine if dict is empty",
        'cell d = new_dict();\nd~udict_set(256, 0, "hello");\nd~udict_set(256, 1, "world");\n\nif (d.dict_empty?()) { ;; Determine if dict is empty\n    ;; dict is empty\n}\nelse {\n    ;; dict is not empty\n}',
    ),
    (
        "How to determine if tuple is empty",
        ';; Declare tlen function because it\'s not presented in stdlib\n(int) tlen (tuple t) asm "TLEN";\n\n() main () {\n    tuple t = empty_tuple();\n    t~tpush(13);\n    t~tpush(37);\n\n    if (t.tlen() == 0) {\n        ;; tuple is empty\n    }\n    else {\n        ;; tuple is not empty\n    }\n}',
    ),
    (
        "How to determine if lisp-style list is empty",
        "tuple numbers = null();\nnumbers = cons(100, numbers);\n\nif (numbers.null?()) {\n    ;; list-style list is empty\n} else {\n    ;; list-style list is not empty\n}",
    ),
    (
        "How to determine a state of the contract is empty",
        ";; `get_data()` will return the data cell from contract state\ncell contract_data = get_data();\nslice cs = contract_data.begin_parse();\n\nif (cs.slice_empty?()) {\n    ;; contract data is empty, so we create counter and save it\n    int counter = 1;\n    ;; create cell, add counter and save in contract state\n    set_data(begin_cell().store_uint(counter, 32).end_cell());\n}\nelse {\n    ;; contract data is not empty, so we get our counter, increase it and save\n    ;; we should specify correct length of our counter in bits\n    int counter = cs~load_uint(32) + 1;\n    set_data(begin_cell().store_uint(counter, 32).end_cell());\n}",
    ),
    (
        "How to build an internal message cell",
        ';; We use literal `a` to get valid address inside slice from string containing address \nslice addr = "EQArzP5prfRJtDM5WrMNWyr9yUTAi0c9o6PfR4hkWy9UQXHx"a;\nint amount = 1000000000;\n;; we use `op` for identifying operations\nint op = 0;\n\ncell msg = begin_cell()\n    .store_uint(0x18, 6)\n    .store_slice(addr)\n    .store_coins(amount)\n    .store_uint(0, 1 + 4 + 4 + 64 + 32 + 1 + 1) ;; default message headers (see sending messages page)\n    .store_uint(op, 32)\n.end_cell();\n\nsend_raw_message(msg, 3); ;; mode 3 - pay fees separately and ignore errors ',
    ),
    (
        "How to contain a body as ref to an internal message cell",
        ';; We use literal `a` to get valid address inside slice from string containing address \nslice addr = "EQArzP5prfRJtDM5WrMNWyr9yUTAi0c9o6PfR4hkWy9UQXHx"a;\nint amount = 1000000000;\nint op = 0;\ncell message_body = begin_cell() ;; Creating a cell with message\n    .store_uint(op, 32)\n    .store_slice("❤")\n.end_cell();\n    \ncell msg = begin_cell()\n    .store_uint(0x18, 6)\n    .store_slice(addr)\n    .store_coins(amount)\n    .store_uint(0, 1 + 4 + 4 + 64 + 32 + 1) ;; default message headers (see sending messages page)\n    .store_uint(1, 1) ;; set bit to 1 to indicate that the cell will go on\n    .store_ref(message_body)\n.end_cell();\n\nsend_raw_message(msg, 3); ;; mode 3 - pay fees separately and ignore errors ',
    ),
    (
        "How to contain a body as slice to an internal message cell",
        ';; We use literal `a` to get valid address inside slice from string containing address \nslice addr = "EQArzP5prfRJtDM5WrMNWyr9yUTAi0c9o6PfR4hkWy9UQXHx"a;\nint amount = 1000000000;\nint op = 0;\nslice message_body = "❤"; \n\ncell msg = begin_cell()\n    .store_uint(0x18, 6)\n    .store_slice(addr)\n    .store_coins(amount)\n    .store_uint(0, 1 + 4 + 4 + 64 + 32 + 1 + 1) ;; default message headers (see sending messages page)\n    .store_uint(op, 32)\n    .store_slice(message_body)\n.end_cell();\n\nsend_raw_message(msg, 3); ;; mode 3 - pay fees separately and ignore errors ',
    ),
    (
        "How to iterate tuples (in both directions)",
        '(int) tlen (tuple t) asm "TLEN";\nforall X -> (tuple) to_tuple (X x) asm "NOP";\n\n() main () {\n    tuple t = to_tuple([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);\n    int len = t.tlen();\n    \n    int i = 0;\n    while (i < len) {\n        int x = t.at(i);\n        ;; do something with x\n        i = i + 1;\n    }\n\n    i = len - 1;\n    while (i >= 0) {\n        int x = t.at(i);\n        ;; do something with x\n        i = i - 1;\n    }\n}',
    ),
    (
        "How to write own functions using asm keyword",
        ';; ~ means it is modifying method\nforall X -> (tuple, X) ~tpop (tuple t) asm "TPOP"; ',
    ),
    (
        "Iterating n-nested tuples",
        'int tuple_length (tuple t) asm "TLEN";\nforall X -> (tuple, X) ~tpop (tuple t) asm "TPOP";\nforall X -> int is_tuple (X x) asm "ISTUPLE";\nforall X -> tuple cast_to_tuple (X x) asm "NOP";\nforall X -> int cast_to_int (X x) asm "NOP";\nforall X -> (tuple) to_tuple (X x) asm "NOP";\n\n;; define global variable\nglobal int max_value;\n\n() iterate_tuple (tuple t) impure {\n    repeat (t.tuple_length()) {\n        var value = t~tpop();\n        if (is_tuple(value)) {\n            tuple tuple_value = cast_to_tuple(value);\n            iterate_tuple(tuple_value);\n        }\n        else {\n            if(value > max_value) {\n                max_value = value;\n            }\n        }\n    }\n}\n\n() main () {\n    tuple t = to_tuple([[2,6], [1, [3, [3, 5]]], 3]);\n    int len = t.tuple_length();\n    max_value = 0; ;; reset max_value;\n    iterate_tuple(t); ;; iterate tuple and find max value\n    ~dump(max_value); ;; 6\n}',
    ),
    (
        "Basic operations with tuples",
        '(int) tlen (tuple t) asm "TLEN";\nforall X -> (tuple, X) ~tpop (tuple t) asm "TPOP";\n\n() main () {\n    ;; creating an empty tuple\n    tuple names = empty_tuple(); \n    \n    ;; push new items\n    names~tpush("Naito Narihira");\n    names~tpush("Shiraki Shinichi");\n    names~tpush("Akamatsu Hachemon");\n    names~tpush("Takaki Yuichi");\n    \n    ;; pop last item\n    slice last_name = names~tpop();\n\n    ;; get first item\n    slice first_name = names.first();\n\n    ;; get an item by index\n    slice best_name = names.at(2);\n\n    ;; getting the length of the list \n    int number_names = names.tlen();\n}',
    ),
    (
        "Resolving type X",
        'forall X -> int is_null (X x) asm "ISNULL";\nforall X -> int is_int (X x) asm "<{ TRY:<{ 0 PUSHINT ADD DROP -1 PUSHINT }>CATCH<{ 2DROP 0 PUSHINT }> }>CONT 1 1 CALLXARGS";\nforall X -> int is_cell (X x) asm "<{ TRY:<{ CTOS DROP -1 PUSHINT }>CATCH<{ 2DROP 0 PUSHINT }> }>CONT 1 1 CALLXARGS";\nforall X -> int is_slice (X x) asm "<{ TRY:<{ SBITS DROP -1 PUSHINT }>CATCH<{ 2DROP 0 PUSHINT }> }>CONT 1 1 CALLXARGS";\nforall X -> int is_tuple (X x) asm "ISTUPLE";\nforall X -> int cast_to_int (X x) asm "NOP";\nforall X -> cell cast_to_cell (X x) asm "NOP";\nforall X -> slice cast_to_slice (X x) asm "NOP";\nforall X -> tuple cast_to_tuple (X x) asm "NOP";\nforall X -> (tuple, X) ~tpop (tuple t) asm "TPOP";\n\nforall X -> () resolve_type (X value) impure {\n    ;; value here is of type X, since we dont know what is the exact value - we would need to check what is the value and then cast it\n    \n    if (is_null(value)) {\n        ;; do something with the null\n    }\n    elseif (is_int(value)) {\n        int valueAsInt = cast_to_int(value);\n        ;; do something with the int\n    }\n    elseif (is_slice(value)) {\n        slice valueAsSlice = cast_to_slice(value);\n        ;; do something with the slice\n    }\n    elseif (is_cell(value)) {\n        cell valueAsCell = cast_to_cell(value);\n        ;; do something with the cell\n    }\n    elseif (is_tuple(value)) {\n        tuple valueAsTuple = cast_to_tuple(value);\n        ;; do something with the tuple\n    }\n}\n\n() main () {\n    ;; creating an empty tuple\n    tuple stack = empty_tuple();\n    ;; let\'s say we have tuple and do not know the exact types of them\n    stack~tpush("Some text");\n    stack~tpush(4);\n    ;; we use var because we do not know type of value\n    var value = stack~tpop();\n    resolve_type(value);\n}',
    ),
    (
        "How to get current time",
        "int current_time = now();\n  \nif (current_time > 1672080143) {\n    ;; do some stuff \n}",
    ),
    (
        "Modulo operations",
        "(int) modulo_operations (int xp, int zp) {  \n   ;; 2^255 - 19 is a prime number for montgomery curves, meaning all operations should be done against its prime\n   int prime = 57896044618658097711785492504343953926634992332820282019728792003956564819949; \n\n   ;; muldivmod handles the next two lines itself\n   ;; int xp+zp = (xp + zp) % prime;\n   ;; int xp-zp = (xp - zp + prime) % prime;\n   (_, int xp+zp*xp-zp) = muldivmod(xp + zp, xp - zp, prime);\n   return xp+zp*xp-zp;\n}",
    ),
    (
        "How to throw errors",
        "int number = 198;\n\nthrow_if(35, number > 50); ;; the error will be triggered only if the number is greater than 50\n\nthrow_unless(39, number == 198); ;; the error will be triggered only if the number is NOT EQUAL to 198\n\nthrow(36); ;; the error will be triggered anyway",
    ),
    (
        "Reversing tuples",
        'forall X -> (tuple, X) ~tpop (tuple t) asm "TPOP";\nint tuple_length (tuple t) asm "TLEN";\nforall X -> (tuple) to_tuple (X x) asm "NOP";\n\n(tuple) reverse_tuple (tuple t1) {\n    tuple t2 = empty_tuple();\n    repeat (t1.tuple_length()) {\n        var value = t1~tpop();\n        t2~tpush(value);\n    }\n    return t2;\n}\n\n() main () {\n    tuple t = to_tuple([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);\n    tuple reversed_t = reverse_tuple(t);\n    ~dump(reversed_t); ;; [10 9 8 7 6 5 4 3 2 1]\n}',
    ),
    (
        "How to remove an item with a certain index from the list",
        'int tlen (tuple t) asm "TLEN";\n\n(tuple, ()) remove_item (tuple old_tuple, int place) {\n    tuple new_tuple = empty_tuple();\n\n    int i = 0;\n    while (i < old_tuple.tlen()) {\n        int el = old_tuple.at(i);\n        if (i != place) {\n            new_tuple~tpush(el);\n        }\n        i += 1;  \n    }\n    return (new_tuple, ());\n}\n\n() main () {\n    tuple numbers = empty_tuple();\n\n    numbers~tpush(19);\n    numbers~tpush(999);\n    numbers~tpush(54);\n\n    ~dump(numbers); ;; [19 999 54]\n\n    numbers~remove_item(1); \n\n    ~dump(numbers); ;; [19 54]\n}',
    ),
    (
        "Determine if slices are equal",
        'int are_slices_equal_1? (slice a, slice b) {\n    return a.slice_hash() == b.slice_hash();\n}\n\nint are_slices_equal_2? (slice a, slice b) asm "SDEQ";\n\n() main () {\n    slice a = "Some text";\n    slice b = "Some text";\n    ~dump(are_slices_equal_1?(a, b)); ;; -1 = true\n\n    a = "Text";\n    ;; We use literal `a` to get valid address inside slice from string containing address\n    b = "EQDKbjIcfM6ezt8KjKJJLshZJJSqX7XOA4ff-W72r5gqPrHF"a;\n    ~dump(are_slices_equal_2?(a, b)); ;; 0 = false\n}',
    ),
    (
        "Determine if cells are equal",
        "int are_cells_equal? (cell a, cell b) {\n    return a.cell_hash() == b.cell_hash();\n}\n\n() main () {\n    cell a = begin_cell()\n            .store_uint(123, 16)\n            .end_cell();\n\n    cell b = begin_cell()\n            .store_uint(123, 16)\n            .end_cell();\n\n    ~dump(are_cells_equal?(a, b)); ;; -1 = true\n}",
    ),
    (
        "Determine if tuples are equal",
        'int tuple_length (tuple t) asm "TLEN";\nforall X -> (tuple, X) ~tpop (tuple t) asm "TPOP";\nforall X -> int cast_to_int (X x) asm "NOP";\nforall X -> cell cast_to_cell (X x) asm "NOP";\nforall X -> slice cast_to_slice (X x) asm "NOP";\nforall X -> tuple cast_to_tuple (X x) asm "NOP";\nforall X -> int is_null (X x) asm "ISNULL";\nforall X -> int is_int (X x) asm "<{ TRY:<{ 0 PUSHINT ADD DROP -1 PUSHINT }>CATCH<{ 2DROP 0 PUSHINT }> }>CONT 1 1 CALLXARGS";\nforall X -> int is_cell (X x) asm "<{ TRY:<{ CTOS DROP -1 PUSHINT }>CATCH<{ 2DROP 0 PUSHINT }> }>CONT 1 1 CALLXARGS";\nforall X -> int is_slice (X x) asm "<{ TRY:<{ SBITS DROP -1 PUSHINT }>CATCH<{ 2DROP 0 PUSHINT }> }>CONT 1 1 CALLXARGS";\nforall X -> int is_tuple (X x) asm "ISTUPLE";\nint are_slices_equal? (slice a, slice b) asm "SDEQ";\n\nint are_cells_equal? (cell a, cell b) {\n    return a.cell_hash() == b.cell_hash();\n}\n\n(int) are_tuples_equal? (tuple t1, tuple t2) {\n    int equal? = -1; ;; initial value to true\n    \n    if (t1.tuple_length() != t2.tuple_length()) {\n        ;; if tuples are differ in length they cannot be equal\n        return 0;\n    }\n\n    int i = t1.tuple_length();\n    \n    while (i > 0 & equal?) {\n        var v1 = t1~tpop();\n        var v2 = t2~tpop();\n        \n        if (is_null(t1) & is_null(t2)) {\n            ;; nulls are always equal\n        }\n        elseif (is_int(v1) & is_int(v2)) {\n            if (cast_to_int(v1) != cast_to_int(v2)) {\n                equal? = 0;\n            }\n        }\n        elseif (is_slice(v1) & is_slice(v2)) {\n            if (~ are_slices_equal?(cast_to_slice(v1), cast_to_slice(v2))) {\n                equal? = 0;\n            }\n        }\n        elseif (is_cell(v1) & is_cell(v2)) {\n            if (~ are_cells_equal?(cast_to_cell(v1), cast_to_cell(v2))) {\n                equal? = 0;\n            }\n        }\n        elseif (is_tuple(v1) & is_tuple(v2)) {\n            ;; recursively determine nested tuples\n            if (~ are_tuples_equal?(cast_to_tuple(v1), cast_to_tuple(v2))) {\n                equal? = 0;\n            }\n        }\n        else {\n            equal? = 0;\n        }\n\n        i -= 1;\n    }\n\n    return equal?;\n}\n\n() main () {\n    tuple t1 = cast_to_tuple([[2, 6], [1, [3, [3, 5]]], 3]);\n    tuple t2 = cast_to_tuple([[2, 6], [1, [3, [3, 5]]], 3]);\n\n    ~dump(are_tuples_equal?(t1, t2)); ;; -1 \n}',
    ),
    (
        "Generate internal address",
        "(slice) generate_internal_address (int workchain_id, cell state_init) {\n    ;; addr_std$10 anycast:(Maybe Anycast) workchain_id:int8 address:bits256  = MsgAddressInt;\n\n    return begin_cell()\n        .store_uint(2, 2) ;; addr_std$10\n        .store_uint(0, 1) ;; anycast nothing\n        .store_int(workchain_id, 8) ;; workchain_id: -1\n        .store_uint(cell_hash(state_init), 256)\n    .end_cell().begin_parse();\n}\n\n() main () {\n    slice deploy_address = generate_internal_address(workchain(), state_init);\n    ;; then we can deploy new contract\n}",
    ),
    (
        "Generate external address",
        '(int) ubitsize (int a) asm "UBITSIZE";\n\nslice generate_external_address (int address) {\n    ;; addr_extern$01 len:(## 9) external_address:(bits len) = MsgAddressExt;\n    \n    int address_length = ubitsize(address);\n    \n    return begin_cell()\n        .store_uint(1, 2) ;; addr_extern$01\n        .store_uint(address_length, 9)\n        .store_uint(address, address_length)\n    .end_cell().begin_parse();\n}',
    ),
    (
        "How to store and load dictionary in local storage",
        "slice local_storage = get_data().begin_parse();\ncell dictionary_cell = new_dict();\nif (~ slice_empty?(local_storage)) {\n    dictionary_cell = local_storage~load_dict();\n}",
    ),
    (
        "How to send a simple message",
        'cell msg = begin_cell()\n    .store_uint(0x18, 6) ;; flags\n    .store_slice("EQBIhPuWmjT7fP-VomuTWseE8JNWv2q7QYfsVQ1IZwnMk8wL"a) ;; destination address\n    .store_coins(100) ;; amount of nanoTons to send\n    .store_uint(0, 1 + 4 + 4 + 64 + 32 + 1 + 1) ;; default message headers (see sending messages page)\n    .store_uint(0, 32) ;; zero opcode - means simple transfer message with comment\n    .store_slice("Hello from FunC!") ;; comment\n.end_cell();\nsend_raw_message(msg, 3); ;; mode 3 - pay fees separately, ignore errors',
    ),
    (
        "How to send a message with an incoming account",
        "() recv_internal (slice in_msg_body) {\n    {-\n        This is a simple example of a proxy-contract.\n        It will expect in_msg_body to contain message mode, body and destination address to be sent to.\n    -}\n\n    int mode = in_msg_body~load_uint(8); ;; first byte will contain msg mode\n    slice addr = in_msg_body~load_msg_addr(); ;; then we parse the destination address\n    slice body = in_msg_body; ;; everything that is left in in_msg_body will be our new message's body\n\n    cell msg = begin_cell()\n        .store_uint(0x18, 6)\n        .store_slice(addr)\n        .store_coins(100) ;; just for example\n        .store_uint(0, 1 + 4 + 4 + 64 + 32 + 1 + 1) ;; default message headers (see sending messages page)\n        .store_slice(body)\n    .end_cell();\n    send_raw_message(msg, mode);\n}",
    ),
    (
        "How to send a message with the entire balance",
        'cell msg = begin_cell()\n    .store_uint(0x18, 6) ;; flags\n    .store_slice("EQBIhPuWmjT7fP-VomuTWseE8JNWv2q7QYfsVQ1IZwnMk8wL"a) ;; destination address\n    .store_coins(0) ;; we don\'t care about this value right now\n    .store_uint(0, 1 + 4 + 4 + 64 + 32 + 1 + 1) ;; default message headers (see sending messages page)\n    .store_uint(0, 32) ;; zero opcode - means simple transfer message with comment\n    .store_slice("Hello from FunC!") ;; comment\n.end_cell();\nsend_raw_message(msg, 128); ;; mode = 128 is used for messages that are to carry all the remaining balance of the current smart contract',
    ),
    (
        "How to send a message with a long text comment",
        '{-\n    If we want to send a message with really long comment, we should split the comment to several slices.\n    Each slice should have <1023 bits of data (127 chars).\n    Each slice should have a reference to the next one, forming a snake-like structure.\n-}\n\ncell body = begin_cell()\n    .store_uint(0, 32) ;; zero opcode - simple message with comment\n    .store_slice("long long long message...")\n    .store_ref(begin_cell()\n        .store_slice(" you can store string of almost any length here.")\n        .store_ref(begin_cell()\n            .store_slice(" just don\'t forget about the 127 chars limit for each slice")\n        .end_cell())\n    .end_cell())\n.end_cell();\n\ncell msg = begin_cell()\n    .store_uint(0x18, 6) ;; flags\n    ;; We use literal `a` to get valid address inside slice from string containing address \n    .store_slice("EQBIhPuWmjT7fP-VomuTWseE8JNWv2q7QYfsVQ1IZwnMk8wL"a) ;; destination address\n    .store_coins(100) ;; amount of nanoTons to send\n    .store_uint(0, 1 + 4 + 4 + 64 + 32 + 1) ;; default message headers (see sending messages page)\n    .store_uint(1, 1) ;; we want to store body as a ref\n    .store_ref(body)\n.end_cell();\nsend_raw_message(msg, 3); ;; mode 3 - pay fees separately, ignore errors',
    ),
    (
        "How to get only data bits from a slice (without refs)",
        'slice s = begin_cell()\n    .store_slice("Some data bits...")\n    .store_ref(begin_cell().end_cell()) ;; some references\n    .store_ref(begin_cell().end_cell()) ;; some references\n.end_cell().begin_parse();\n\nslice s_only_data = s.preload_bits(s.slice_bits());',
    ),
    (
        "How to define your own modifying method",
        '(slice, (int)) load_digit (slice s) {\n    int x = s~load_uint(8); ;; load 8 bits (one char) from slice\n    x -= 48; ;; char \'0\' has code of 48, so we substract it to get the digit as a number\n    return (s, (x)); ;; return our modified slice and loaded digit\n}\n\n() main () {\n    slice s = "258";\n    int c1 = s~load_digit();\n    int c2 = s~load_digit();\n    int c3 = s~load_digit();\n    ;; here s is equal to "", and c1 = 2, c2 = 5, c3 = 8\n}',
    ),
    (
        "How to raise number to the power of n",
        ";; Unoptimized variant\nint pow (int a, int n) {\n    int i = 0;\n    int value = a;\n    while (i < n - 1) {\n        a *= value;\n        i += 1;\n    }\n    return a;\n}\n\n;; Optimized variant\n(int) binpow (int n, int e) {\n    if (e == 0) {\n        return 1;\n    }\n    if (e == 1) {\n        return n;\n    }\n    int p = binpow(n, e / 2);\n    p *= p;\n    if ((e % 2) == 1) {\n        p *= n;\n    }\n    return p;\n}\n\n() main () {\n    int num = binpow(2, 3);\n    ~dump(num); ;; 8\n}",
    ),
    (
        "How to convert string to int",
        'slice string_number = "26052021";\nint number = 0;\n\nwhile (~ string_number.slice_empty?()) {\n    int char = string_number~load_uint(8);\n    number = (number * 10) + (char - 48); ;; we use ASCII table\n}\n\n~dump(number);',
    ),
    (
        "How to convert int to string",
        "int n = 261119911;\nbuilder string = begin_cell();\ntuple chars = null();\ndo {\n    int r = n~divmod(10);\n    chars = cons(r + 48, chars);\n} until (n == 0);\ndo {\n    int char = chars~list_next();\n    string~store_uint(char, 8);\n} until (null?(chars));\n\nslice result = string.end_cell().begin_parse();\n~dump(result);",
    ),
    (
        "How to iterate dictionaries",
        'cell d = new_dict();\nd~udict_set(256, 1, "value 1");\nd~udict_set(256, 5, "value 2");\nd~udict_set(256, 12, "value 3");\n\n;; iterate keys from small to big\n(int key, slice val, int flag) = d.udict_get_min?(256);\nwhile (flag) {\n    ;; do something with pair key->val\n    \n    (key, val, flag) = d.udict_get_next?(256, key);\n}',
    ),
    (
        "How to delete value from dictionaries",
        'cell names = new_dict();\nnames~udict_set(256, 27, "Alice");\nnames~udict_set(256, 25, "Bob");\n\nnames~udict_delete?(256, 27);\n\n(slice val, int key) = names.udict_get?(256, 27);\n~dump(val); ;; null() -> means that key was not found in a dictionary',
    ),
    (
        "How to iterate cell tree recursively",
        'forall X -> int is_null (X x) asm "ISNULL";\nforall X -> (tuple, ()) push_back (tuple tail, X head) asm "CONS";\nforall X -> (tuple, (X)) pop_back (tuple t) asm "UNCONS";\n\n() main () {\n    ;; just some cell for example\n    cell c = begin_cell()\n        .store_uint(1, 16)\n        .store_ref(begin_cell()\n            .store_uint(2, 16)\n        .end_cell())\n        .store_ref(begin_cell()\n            .store_uint(3, 16)\n            .store_ref(begin_cell()\n                .store_uint(4, 16)\n            .end_cell())\n            .store_ref(begin_cell()\n                .store_uint(5, 16)\n            .end_cell())\n        .end_cell())\n    .end_cell();\n\n    ;; creating tuple with no data, which plays the role of stack\n    tuple stack = null();\n    ;; bring the main cell into the stack to process it in the loop\n    stack~push_back(c);\n    ;; do it until stack is not null\n    while (~ stack.is_null()) {\n        ;; get the cell from the stack and convert it to a slice to be able to process it\n        slice s = stack~pop_back().begin_parse();\n\n        ;; do something with s data\n\n        ;; if the current slice has any refs, add them to stack\n        repeat (s.slice_refs()) {\n            stack~push_back(s~load_ref());\n        }\n    }\n}',
    ),
    (
        "How to iterate through lisp-style list",
        'forall X -> int is_null (X x) asm "ISNULL";\nforall X -> (tuple, ()) push_back (tuple tail, X head) asm "CONS";\nforall X -> (tuple, (X)) pop_back (tuple t) asm "UNCONS";\n\n() main () {\n    ;; some example list\n    tuple l = null();\n    l~push_back(1);\n    l~push_back(2);\n    l~push_back(3);\n\n    ;; iterating through elements\n    ;; note that this iteration is in reversed order\n    while (~ l.is_null()) {\n        var x = l~pop_back();\n\n        ;; do something with x\n    }\n}',
    ),
    (
        "How to send a deploy message (with stateInit only, with stateInit and body)",
        "() deploy_with_stateinit(cell message_header, cell state_init) impure {\n  var msg = begin_cell()\n    .store_slice(begin_parse(msg_header))\n    .store_uint(2 + 1, 2) ;; init:(Maybe (Either StateInit ^StateInit))\n    .store_uint(0, 1) ;; body:(Either X ^X)\n    .store_ref(state_init)\n    .end_cell();\n\n  ;; mode 64 - carry the remaining value in the new message\n  send_raw_message(msg, 64); \n}\n\n() deploy_with_stateinit_body(cell message_header, cell state_init, cell body) impure {\n  var msg = begin_cell()\n    .store_slice(begin_parse(msg_header))\n    .store_uint(2 + 1, 2) ;; init:(Maybe (Either StateInit ^StateInit))\n    .store_uint(1, 1) ;; body:(Either X ^X)\n    .store_ref(state_init)\n    .store_ref(body)\n    .end_cell();\n\n  ;; mode 64 - carry the remaining value in the new message\n  send_raw_message(msg, 64); \n}",
    ),
    (
        "How to build a stateInit cell",
        "() build_stateinit(cell init_code, cell init_data) {\n  var state_init = begin_cell()\n    .store_uint(0, 1) ;; split_depth:(Maybe (## 5))\n    .store_uint(0, 1) ;; special:(Maybe TickTock)\n    .store_uint(1, 1) ;; (Maybe ^Cell)\n    .store_uint(1, 1) ;; (Maybe ^Cell)\n    .store_uint(0, 1) ;; (HashmapE 256 SimpleLib)\n    .store_ref(init_code)\n    .store_ref(init_data)\n    .end_cell();\n}",
    ),
    (
        "How to calculate a contract address (using stateInit)",
        "() calc_address(cell state_init) {\n  var future_address = begin_cell() \n    .store_uint(2, 2) ;; addr_std$10\n    .store_uint(0, 1) ;; anycast:(Maybe Anycast)\n    .store_uint(0, 8) ;; workchain_id:int8\n    .store_uint(cell_hash(state_init), 256) ;; address:bits256\n    .end_cell();\n}",
    ),
]

base_prompt = """
The FunC programming language, utilized for writing smart contracts in the TON blockchain, is characterized by a distinct structure and syntax. Here's a summary covering its various aspects:

Basic Syntax and Operations
Conditional Statements: FunC uses an if statement similar to other languages, but with true as -1 and false as 0. There's no need for the == operator for comparison to 0​​.
Loop Constructs:
Repeat Loop: For iterating a set number of times, the repeat loop is used​​.
While Loop: This is used when the number of iterations isn't known in advance​​.
Do Until Loop: Runs at least once and continues until a condition is met​​.
Working with Slices and Cells
Slices: Used to check if they're empty or contain data and references​​.
Cells: To determine if a cell is empty, it's first converted to a slice, then checked using slice_empty?(), slice_data_empty?(), or slice_data_refs?()​​.
Data Structures and State Management
Dictionaries: FunC checks for emptiness in dictionaries using dict_empty?()​​.
Tuples: It's important to know if tuples are empty to avoid extraction errors​​.
List-Style Lists: The presence or absence of elements in list-style lists is checked using null?()​​.
Contract State: Determining if a contract's state is empty is crucial for initializing state variables​​.
Message Handling
Internal Message Cells: These are created by specifying technical flags, recipient address, and data​​.
Message Body: It can be included as a reference or slice in the internal message cell​​.
Iteration and Custom Functions
Iterating Tuples: Tuples can be iterated in both directions using a combination of custom and built-in functions​​.
Custom Functions: FunC allows the creation of custom functions using the asm keyword, offering more flexibility and control​​.
Nested Tuples: Iterating nested tuples requires handling both tuples and individual elements within them​​.
Types and Polymorphism
Built-In Types: FunC includes types like int, cell, slice, builder, tuple, and cont​​.
Boolean Representation: Booleans are represented as integers (false as 0, true as -1) and logical operations are performed bitwise​​.
Null Values: null represents the absence of a value, and default handling leads to a runtime exception​​.
Type Inference: Supports type inference with type "holes" (_ and var) for later resolution during type checking​​.
Composite Types: FunC allows the composition of complex types, like functional types, tensor types, and unit types​​.
Polymorphism: It employs a Miller-Rabin type system supporting polymorphic functions, using type variables​​.
This summary provides an overview of FunC's structure and features, which can be used to generate sample code and understand its syntax and semantics.

Use ;; or {- -} to comment your code.

################### EXAMPLES ###################

;; If Statement
int flag = 0; // false
if (flag) {
    // do something
} else {
    // reject transaction
}

;; Repeat Loop
int number = 2;
repeat(4) {
    number *= 2;
}

;; While Loop
slice msg = message.begin_parse();
while (msg.slice_refs_empty?() != -1) {
    // process msg
}

;; Do Until Loop
int flag = 0;
do {
    // do something
} until (flag == -1);

;; Check if Slice is Empty
slice empty_slice = "";
if (empty_slice.slice_empty?()) {
    // slice is empty
}

;; Check if Dict is Empty
cell d = new_dict();
if (d.dict_empty?()) {
    // dict is empty
}

;; Check if Tuple is Empty
tuple t = empty_tuple();
if (t.tlen() == 0) {
    // tuple is empty
}

;; Internal Message Cell
cell msg = begin_cell()
    .store_uint(0x18, 6)
    .store_slice(addr)
    .store_coins(amount)
    .end_cell();

############### MORE EXAMPLES ##############
<examples>
################### TASK ###################

Here are a documentation and some examples of working with FUNC language.
Write your own example based on this language structure. Make it differ from original text.
Use some of the following keywords: <keywords>.
No needs to make it correct. Just make it look like a real code.
Write just a code without explanation and make it very natural.
WRAP ALL COMMANDS WITH TRIPLE BACKTICKS LIKE THESE:``` ```.
DON'T USE COMMENTS IN THE CODE. DON'T COMMENT THE CODE. ANSWER WITH ONE CODE BLOCK.
"""


def extract_code(text):
    code_blocks = re.findall(r"```.*?```", text, re.DOTALL)
    if not code_blocks:
        return None
    all_code_blocks = []
    for i, code_block in enumerate(code_blocks):
        lines = code_block.splitlines()
        lines[0] = lines[0].replace("```", "")
        if len(lines[0]) < 15:
            lines = lines[1:]
        lines[-1] = lines[-1].replace("```", "")
        all_code_blocks.append("\n".join(lines))

    code = "\n".join(all_code_blocks)
    if "int main" in code:
        return None

    code = code.replace("\n\n", "\n")
    return code


def ask_llama(prompt, context=None):
    url = "http://localhost:11433/api/generate"

    req = {"model": "llama2:13b", "stream": False, "prompt": prompt}
    if context is not None:
        req["context"] = context

    while True:
        try:
            response = requests.request(
                "POST", url, data=json.dumps(req), timeout=60
            ).json()
            break
        except:
            continue

    return response["response"], response["context"]


def get_code(limit=1000, offset=0, save_root=None):
    lang_name = "FUNC"
    constructs = utils.lang_constructs.lang_keywords[lang_name]
    all_kws = constructs

    all_requests = []
    for i in range(limit):
        keywords = ", ".join([f'"{s}"' for s in random.sample(all_kws, 2)])
        examples = random.sample(ton_examples, 3)
        examples = "\n".join([f";; {e[0]}\n{e[1]}" for e in examples])

        request = base_prompt
        request.replace("<examples>", examples).replace("<keywords>", keywords)
        all_requests.append(request)

    if save_root is not None:
        save_dir = os.path.join(save_root, lang_name)
        os.makedirs(save_dir, exist_ok=True)

    code_blocks = []

    for i, prompt in enumerate(
        tqdm.tqdm(all_requests, desc="Processing language '%s'" % lang_name)
    ):
        real_i = i + offset
        if save_root is not None:
            request_save_dir = os.path.join(save_dir, str(real_i))
            if os.path.exists(request_save_dir):
                continue

            os.makedirs(request_save_dir, exist_ok=True)

        response, _ = ask_llama(prompt)

        code = extract_code(response)
        if code is None:
            continue
        code_blocks.append(code)

        if save_root is not None:
            with open(os.path.join(request_save_dir, "0.txt"), "w") as f:
                f.write(code)

    return code_blocks


if __name__ == "__main__":
    res = get_code(limit=4000, offset=4000, save_root="../../datasets/llama/tasks2/")
