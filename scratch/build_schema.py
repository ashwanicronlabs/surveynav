"""Build the complete synthetic Confirmit survey schema from client specifications."""
import json
from pathlib import Path

def get_full_schema():
    languages = [
        {"confirmitLanguageId": 9},
        {"confirmitLanguageId": 18},
        {"confirmitLanguageId": 31},
        {"confirmitLanguageId": 1046},
        {"confirmitLanguageId": 2058}
    ]

    # Helper builders
    def num(name, lower=0.0, upper=100.0, is_compound=True, fields=None, title="", is_hidden=False):
        res = {
            "name": name,
            "variableType": "numeric",
            "isCompound": is_compound,
            "isBackground": False,
            "isHidden": is_hidden,
            "isSystemVariable": False,
            "isUserVariable": False,
            "precision": 2,
            "scale": 2,
            "excludeFromReporting": False
        }
        if lower is not None:
            res["lowerNumericLimit"] = lower
            res["lowerLimitType"] = "inclusive"
        if upper is not None:
            res["upperNumericLimit"] = upper
            res["upperLimitType"] = "inclusive"
        if fields:
            res["fields"] = [{"code": str(c), "hasOtherField": False} for c in fields]
        elif is_compound:
            res["fields"] = [{"code": "1", "hasOtherField": False}]
        return res

    def single(name, options, is_hidden=False, is_bg=False, is_compound=False, fields=None):
        opts = []
        for o in options:
            if isinstance(o, tuple):
                code, txt = o
                opts.append({"code": str(code), "hasOtherField": False, "texts": [{"languageId": 1, "text": txt}]})
            else:
                opts.append({"code": str(o), "hasOtherField": False, "texts": [{"languageId": 1, "text": f"Option {o}"}]})
        res = {
            "name": name,
            "variableType": "singleChoice",
            "isCompound": is_compound,
            "isBackground": is_bg,
            "isHidden": is_hidden,
            "isSystemVariable": False,
            "isUserVariable": False,
            "options": opts,
            "excludeFromReporting": False
        }
        if fields:
            res["fields"] = [{"code": str(f), "hasOtherField": False} for f in fields]
        return res

    def multi(name, options, is_hidden=False, is_bg=False):
        opts = []
        for o in options:
            if isinstance(o, tuple):
                code, txt = o
                opts.append({"code": str(code), "hasOtherField": False, "texts": [{"languageId": 1, "text": txt}]})
            else:
                opts.append({"code": str(o), "hasOtherField": False, "texts": [{"languageId": 1, "text": f"Option {o}"}]})
        return {
            "name": name,
            "variableType": "multiChoice",
            "isCompound": False,
            "isBackground": is_bg,
            "isHidden": is_hidden,
            "isSystemVariable": False,
            "isUserVariable": False,
            "options": opts,
            "excludeFromReporting": False
        }

    def text_var(name, is_hidden=False, is_bg=False, is_compound=False, length=None):
        res = {
            "name": name,
            "variableType": "text",
            "isCompound": is_compound,
            "isBackground": is_bg,
            "isHidden": is_hidden,
            "isSystemVariable": False,
            "isUserVariable": False,
            "excludeFromReporting": False
        }
        if length is not None:
            res["length"] = length
        if is_compound:
            res["fields"] = [{"code": "1", "hasOtherField": False}]
        return res

    def dt(name):
        return {
            "name": name,
            "variableType": "dateTime",
            "isCompound": False,
            "isBackground": False,
            "isHidden": False,
            "isSystemVariable": True,
            "isUserVariable": False,
            "excludeFromReporting": False
        }

    def info(name):
        return {
            "name": name,
            "variableType": "info",
            "isCompound": False,
            "isBackground": False,
            "isHidden": False,
            "isSystemVariable": False,
            "isUserVariable": False,
            "excludeFromReporting": False
        }

    # Root variables
    variables = [
        {"name": "respid", "variableType": "numeric", "isCompound": False, "isBackground": False, "isHidden": False, "isSystemVariable": True, "isUserVariable": False, "excludeFromReporting": False},
        {"name": "status", "variableType": "singleChoice", "isCompound": False, "isBackground": False, "isHidden": False, "isSystemVariable": True, "isUserVariable": False, "options": [{"code": c, "hasOtherField": False} for c in ["complete", "incomplete", "quotafull", "error", "screened"]], "excludeFromReporting": False},
        dt("interview_start"),
        dt("interview_end"),
        single("lastchannel", [1, 2, 8, 4]),
        single("lastdevicetype", [1, 2, 8]),
        single("lastrenderingmode", [1, 2, 8]),
        dt("last_touched"),
        dt("lastcomplete"),
        text_var("first_question_on_last_page_displayed", length=2),
        text_var("hiduserid", is_hidden=True),
        text_var("hidredirectlink", is_hidden=True),
        single("hPunch", [1, 2, 3, 4], is_bg=True),
        text_var("bgrpid", is_bg=True),
        single("bgSource", [1, 2, 3, 4], is_bg=True),
        single("hidsource", [1, 2, 3, 4], is_hidden=True),
        single("bgtest", [1, 2], is_bg=True),
        single("isTest", [1, 2], is_hidden=True),
        single("bgLinkStatus", [1, 2], is_bg=True),
        text_var("hidrespurl", is_hidden=True),
        text_var("hidquerystring", is_hidden=True),
        single("l", [1046, 2058, 31, 18], is_hidden=True),
        multi("ErrorMessage", [1, 2, 3, 6, 7, 8, 9, 10, 11, 12], is_hidden=True),
        single("S0", [1, 2, 3, 4], is_hidden=True),
        info("INTRODUCTION"),
        info("Privacy"),
        single("S0a", [1, 2]),
        single("S0b", [1, 2]),
        single("S0c", [1, 2]),
        multi("S0d", [1, 2]),
        single("S1", [1, 2]),
        single("S2", [1, 2]),
        single("S3", [1, 2]),
        single("S4", [1, 2, 3]),
        num("S5", 0.0, 100.0), # planted fault F04 touches S5 range!
        single("S6", [1, 2, 3, 4, 5, 6, 7, 8]),
        num("S7", 0.0, 100.0),
        num("S8", 0.0, 100.0),
        num("S9", None, None),
        num("S10", None, None, fields=[1, 2, 3, 4, 96]),
        single("S11", [1, 2]),
        single("S12", [1, 2]),
        num("S13", None, None),
        single("S14", [1, 2]),
        num("tSection3", None, None, is_compound=False),
        num("Q1", None, None),
        num("Q2", None, None),
        single("Q3", [1, 2, 3, 4, 5, 6, 7]),
        num("Q4", 0.0, 100.0, fields=[1, 2, 3, 4]),
        single("Q5", [1, 2, 96]),
        num("Q6", 0.0, 100.0),
        info("introQ7"),
        num("Q7Mx", 0.0, 100.0, fields=[1, 2, 3]),
        num("Q7Sk", 0.0, 100.0, fields=[1, 2, 3, 4]), # planted fault F01!
        num("Q7Tk", 0.0, 100.0, fields=[1, 2, 3, 4]),
        multi("Q7Dk", [99]),
        single("Q9", [1, 2, 3, 4, 5]),
        multi("Q10", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 96]),
        single("Q11", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 99]),
        num("Q12", 0.0, 100.0, fields=[1, 2, 3, 4, 5, 6, 7]),
        single("Q17", [1, 2, 3]),
        single("Q18", [1, 2, 3, 4, 5, 6, 7], is_compound=True, fields=[1]),
        single("Q19", [1, 2, 3, 4, 5, 6, 7], is_compound=True, fields=[1, 2, 3]),
        num("Q20", 0.0, 100.0, fields=[1, 2, 3, 4, 5, 6, 7, 8, 96]),
        num("Q21", 0.0, 100.0),
        single("T11", [1, 2, 3]),
        single("T12", [1, 2, 3, 4]),
        {"name": "T24New", "variableType": "ranking", "isCompound": True, "isBackground": False, "isHidden": False, "isSystemVariable": False, "isUserVariable": False, "fields": [{"code": str(c), "hasOtherField": False} for c in [1, 2, 3, 4, 5, 6, 7, 8, 96]], "excludeFromReporting": False},
        single("T13", [1, 2, 3, 4, 5, 6, 7], is_compound=True, fields=[1, 2, 3, 4, 5, 6, 7, 8, 96]),
        single("T14", [1, 2], is_compound=True, fields=[1, 2, 3, 4, 5, 6, 7]), # straightlining planted fault F02!
        num("T15", 0.0, 100.0, fields=[1, 2, 3]),
        num("T16", 0.0, 100.0),
        single("T17", [1, 2, 3, 4]),
        num("T18", 0.0, 100.0),
        multi("T18Dk", [1]),
        multi("T19", [1, 2, 3, 4, 5, 6, 7, 96]),
        num("T20", None, None, fields=[1, 2, 3]),
        num("T21", 0.0, 100.0, fields=[1, 2, 3]),
        num("T23", 0.0, 100.0),
        single("T24", [1, 2, 3, 4]),
        single("hAPPENDIX", [1, 2, 3, 4]),
        single("PCurrencyHidden", [1, 2, 3, 4], is_hidden=True),
        info("PRICING_INTRO"),
        num("P1", 0.0, None, fields=[1, 2, 3, 4]),
        num("P2", 0.0, None),
        single("P3", [1, 2]),
        single("PP1", [1, 2, 3, 4]),
        single("PP2", [1, 2, 3, 4]),
        single("PP3", [1, 2, 3, 4], is_hidden=True),
        single("PP4", [1, 2, 3, 4], is_hidden=True),
        single("PP5", [1, 2, 3, 4], is_hidden=True),
        multi("dPSPriceShown", [1, 2, 3, 4, 5], is_hidden=True),
        single("dPSAllPunch", [1, 2, 3, 4, 5], is_hidden=True, is_compound=True, fields=[1, 2, 3, 4, 5]),
        single("dPSPriceOrder", [1, 2, 3, 4, 5], is_hidden=True, is_compound=True, fields=[1, 2, 3, 4, 5]),
        num("P9", 0.0, None),
        num("P10", 0.0, 100.0),
        num("P11", 0.0, 100.0),
    ]

    for i in range(1, 11):
        variables.append(text_var(f"P12x{i}", is_compound=True))
    variables.append(multi("P12Ex", [1]))
    variables.append({"name": "P12", "variableType": "text", "isCompound": True, "isBackground": False, "isHidden": True, "isSystemVariable": False, "isUserVariable": False, "fields": [{"code": str(c), "hasOtherField": False} for c in range(1, 11)], "excludeFromReporting": False})
    variables.append(info("infoEND"))
    variables.append(num("tSectionTotal", None, None, is_compound=False, is_hidden=True))
    variables.append(text_var("hidQueDropout", is_hidden=True))
    variables.append(single("hidQualified", [1, 2], is_hidden=True))
    variables.append(multi("hidCustomErrorMsg", ["oe", "oes", "generic1", "oealpha", "numrng", "regcity", "oewords", "num100cnt", "num100pct", "DateDKNA", "DateDOB1", "DateFuture1", "DateFuture1Row", "DateDiagFuture", "DateB4Diag", "DateB4DiagRow", "numrngcust", "numwhole", "DateDiffAfter", "DateDiffBefore"], is_hidden=True))
    variables.append(multi("hidCustomLabels", ["s4", "s6", "s7", "s8", "s9", "s10", "s10a", "m10", "m11"], is_hidden=True))
    variables.append(multi("hidFlatlineFlag", ["m0", "m9", "q4", "q5"], is_hidden=True))
    variables.append(num("hidLOI", None, None, is_compound=False, is_hidden=True))
    variables.append(num("hidLOISec", None, None, is_compound=False, is_hidden=True))
    variables.append(text_var("hidSurveyStart", is_hidden=True))
    variables.append(text_var("hidSurveyEnd", is_hidden=True))
    variables.append(text_var("tStart", is_hidden=True))
    variables.append(text_var("tEnd", is_hidden=True))
    variables.append(text_var("hBrowserType", is_hidden=True))
    variables.append(text_var("hBrowserVersion", is_hidden=True))
    variables.append(text_var("hBrowser", is_hidden=True))
    variables.append(text_var("hIPAddress", is_hidden=True, length=2))
    variables.append(single("hIPFlag", [1, 2], is_hidden=True))
    variables.append(text_var("hScreenResolution", is_hidden=True))
    variables.append(single("hSRMode", [1, 2, 3, 4], is_hidden=True))
    variables.append(single("hDeviceTypeMain", [1, 2, 3, 4], is_hidden=True))
    variables.append(text_var("hUserAgent", is_hidden=True))
    variables.append(single("hOperatingSystem", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 21, 22, 23, 24, 25, 26, 27, 28, 99], is_hidden=True))
    variables.append(single("hDeviceType", [1, 2, 3, 4, 5, 6, 7, 8, 9], is_hidden=True))
    variables.append(info("iComplete"))
    variables.append(multi("hidScreened", ["uid", "S1", "S2", "S3", "S4", "S6", "S11", "S12", "S14", "S5", "S7", "S9", "S10", "S13", "S8", "S0a", "S0"], is_hidden=True))
    variables.append(single("hidScreenedFirst", ["uid", "S1", "S2", "S3", "S4", "S6", "S11", "S12", "S14", "S5", "S7", "S9", "S10", "S13", "S8", "S0a", "S0"], is_hidden=True))
    variables.append(single("hidScreenedHold", [1, 2], is_hidden=True))
    variables.append(single("hidScreenedImm", [1, 2], is_hidden=True))
    variables.append(info("iScreenoutImm"))
    variables.append(info("iScreenoutSec"))
    variables.append(multi("hidquotafull", ["tot", "totpost", "s2", "s2post", "mdd", "S4", "S0"], is_hidden=True))
    variables.append(info("iQF"))

    # Children loops: LoopT1 and LpPSPrices
    loop_t1_vars = [info("Q4_INTRO"), info("TPPIntro")]
    for i in range(1, 11):
        loop_t1_vars.append(text_var(f"T1x{i}", is_compound=True))
    loop_t1_vars.append(multi("T1Ex", [1]))
    loop_t1_vars.append({"name": "T1", "variableType": "text", "isCompound": True, "isBackground": False, "isHidden": True, "isSystemVariable": False, "isUserVariable": False, "fields": [{"code": str(c), "hasOtherField": False} for c in range(1, 11)], "excludeFromReporting": False})
    for i in range(1, 11):
        loop_t1_vars.append(text_var(f"T2x{i}", is_compound=True))
    loop_t1_vars.append(multi("T2Ex", [1]))
    loop_t1_vars.append({"name": "T2", "variableType": "text", "isCompound": True, "isBackground": False, "isHidden": True, "isSystemVariable": False, "isUserVariable": False, "fields": [{"code": str(c), "hasOtherField": False} for c in range(1, 11)], "excludeFromReporting": False})
    loop_t1_vars.append(single("T3", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], is_compound=True, fields=[1]))
    loop_t1_vars.append(single("T4", [1, 2, 3, 4]))
    loop_t1_vars.append(num("T5", 0.0, 100.0))
    loop_t1_vars.append(multi("T6", [1, 2, 3, 4, 5, 6, 7]))
    loop_t1_vars.append(single("T7", [1, 2, 3, 4, 5, 6, 7], is_compound=True, fields=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 96]))
    loop_t1_vars.append(single("T7b", [1, 2]))
    loop_t1_vars.append(single("T9", [1, 2]))
    for i in range(1, 11):
        loop_t1_vars.append(text_var(f"T10x{i}", is_compound=True))
    loop_t1_vars.append(multi("T10Ex", [1]))
    loop_t1_vars.append({"name": "T10", "variableType": "text", "isCompound": True, "isBackground": False, "isHidden": True, "isSystemVariable": False, "isUserVariable": False, "fields": [{"code": str(c), "hasOtherField": False} for c in range(1, 11)], "excludeFromReporting": False})

    loop_t1 = {
        "name": "LoopT1",
        "keys": [{"name": "LoopT1", "variableType": "singleChoice", "isCompound": False, "isBackground": False, "isHidden": False, "isSystemVariable": False, "isUserVariable": False, "options": [{"code": "1"}, {"code": "2"}], "excludeFromReporting": False}],
        "variables": loop_t1_vars
    }

    loop_prices_vars = [
        single("dLpPSItr", [1, 2, 3, 4, 5]),
        single("P6", [1, 2, 3, 4, 5]),
        num("P7", 0.0, 100.0),
        multi("P8", [1, 2, 3, 4, 5, 6, 7, 96]),
        single("dPSResponseType", [1, 2], is_hidden=True)
    ]
    loop_prices = {
        "name": "LpPSPrices",
        "keys": [{"name": "LpPSPrices", "variableType": "singleChoice", "isCompound": False, "isBackground": False, "isHidden": False, "isSystemVariable": False, "isUserVariable": False, "options": [{"code": str(c)} for c in range(1, 6)], "excludeFromReporting": False}],
        "variables": loop_prices_vars
    }

    schema = {
        "defaultConfirmitLanguageId": 1,
        "languages": languages,
        "root": {
            "name": "response",
            "texts": [{"languageId": 1, "text": "Please review this section before continuing. [Demo response; schema item 855]"}],
            "keys": [{"name": "responseid", "variableType": "numeric", "isCompound": False, "isBackground": False, "isHidden": False, "isSystemVariable": True, "isUserVariable": False, "excludeFromReporting": False}],
            "variables": variables,
            "children": [loop_t1, loop_prices]
        },
        "links": {"self": "https://example.invalid/synthetic/schema"}
    }
    return schema

if __name__ == "__main__":
    schema = get_full_schema()
    out_path = Path("/home/cron/SurveyNav/data/synthetic_survey_responses_schema.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(schema, f, indent=2)
    
    # Count total variables
    root_vars = len(schema["root"]["variables"])
    t1_vars = len(schema["root"]["children"][0]["variables"])
    price_vars = len(schema["root"]["children"][1]["variables"])
    total_vars = root_vars + t1_vars + price_vars
    print(f"Schema created successfully!")
    print(f"Root variables: {root_vars}, LoopT1 variables: {t1_vars}, LpPSPrices variables: {price_vars}")
    print(f"Total variables defined: {total_vars}")
