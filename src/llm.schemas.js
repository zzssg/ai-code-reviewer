export const architecture_review_schema = {
    type: "object",
    properties: {
        summary: {
            type: "string",
            description: "Architectural understanding of the changes introduced by this pull-request. Group filenames by the change and explicitly mention these groups",
        },
        movements: {
            type: "array",
            description: "Detected structural movements across the pull-request",
            items: {
                type: "object",
                properties: {
                    type: {
                        type: "string",
                        description: "Movement type",
                        enum: ["code_move", "rename", "deletion", "split", "merge", "unclear"]
                    },
                    from: {
                        type: "array",
                        description: "Source paths involved in the movement",
                        items: { type: "string" }
                    },
                    to: {
                        type: "array",
                        description: "Destination paths involved in the movement",
                        items: { type: "string" }
                    },
                    confidence: {
                        type: "string",
                        description: "Confidence level for this movement detection",
                        enum: ["high", "medium", "low"]
                    },
                    reasoning: {
                        type: "string",
                        description: "Brief explanation based only on provided diffs"
                    }
                },
                required: ["type", "from", "to", "confidence", "reasoning"]
            }
        },
        review_buckets: {
            type: "array",
            description: "Logical groups of files to review together",
            items: {
                type: "object",
                properties: {
                    name: {
                        type: "string",
                        description: "Short bucket name"
                    },
                    goal: {
                        type: "string",
                        description: "What should be reviewed together"
                    },
                    files: {
                        type: "array",
                        description: "File paths belonging to this review bucket",
                        items: { type: "string" }
                    },
                    rationale: {
                        type: "string",
                        description: "Why these files belong together"
                    }
                },
                required: ["name", "goal", "files", "rationale"]
            }
        },
    },
    required: ["summary", "movements", "review_buckets"]
};


export const pr_review_issues_schema = {
    type: "object",
    properties: {
        potential_issues: {
            type: "array",
            items: {
                type: "object",
                properties: {
                    issue_description: { type: "string", description: "Description of potential problem, edge cases or missing considerations" },
                    severity: {
                        type: "string",
                        description: "Severity of the issue calculated by the LLM",
                        enum: ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
                    },
                    locations: {
                        type: "array",
                        description: "All file locations where this issue occurs",
                        items: {
                            type: "object",
                            properties: {
                                filename: { type: "string", description: "Name of file this finding is addressed for" },
                                line_number: { type: "number", description: "Exact line number from the diff prefix (the N in +N:, -N:, or  N:)" },
                                line_type: {
                                    type: "string",
                                    enum: ["ADDED", "REMOVED", "CONTEXT"],
                                    description: "ADDED for lines prefixed with +N:, REMOVED for lines prefixed with -N:, CONTEXT for lines prefixed with  N: (space)"
                                }
                            },
                            required: ["filename", "line_number", "line_type"]
                        }
                    },
                    suggestion: {
                        type: "object",
                        properties: {
                            code: {
                                type: "string",
                                description: "Formatted multi-line code block implementing suggestion on how to fix the issue. Put only real working code here. Wrap it with triple backticks and specify language if possible."
                            },
                            text: {
                                type: "string",
                                description: "Short plain text description of the suggested action. No code should be provided here."
                            }
                        }
                    }
                },
                required: ["issue_description", "severity", "locations", "suggestion"]
            },
            description: "List of inline review findings for the file(s) under review"
        }
    },
    required: ["potential_issues"]
};

export const pr_summary_schema = {
    type: "object",
    properties: {
        summary: { type: "string", description: "Concise overall summary of the PR review" },
        potential_issues: {
            type: "array",
            items: {
                type: "object",
                properties: {
                    issue_description: { type: "string", description: "Description of potential problem, edge cases or missing considerations" },
                    severity: {
                        type: "string",
                        description: "Severity of the issue calculated by the LLM",
                        enum: ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
                    },
                    locations: {
                        type: "array",
                        description: "All file locations where this issue occurs",
                        items: {
                            type: "object",
                            properties: {
                                filename: { type: "string", description: "Name of file this finding is addressed for" },
                                line_number: { type: "number", description: "Filename line number where this finding occurs" }
                            },
                            required: ["filename", "line_number"]
                        }
                    },
                    suggestion: {
                        type: "object",
                        properties: {
                            code: {
                                type: "string",
                                description: "Formatted multi-line code block implementing suggestion on how to fix the issue. Put only real working code here. Wrap it with triple backticks and specify language if possible."
                            },
                            text: {
                                type: "string",
                                description: "Short plain text description of the suggested action. No code should be provided here."
                            }
                        }
                    }
                },
                required: ["issue_description", "severity", "locations", "suggestion"]
            },
            description: "Merged and deduplicated issues across all reviewed files"
        },
        verdict: { type: "string", description: "Overall PR verdict represented with one of values: Approve or Request-changes and a short reason for this verdict. This field is optional and may be omitted if the LLM cannot determine a clear verdict based on the review findings." }
    },
    required: ["summary", "potential_issues"]
};