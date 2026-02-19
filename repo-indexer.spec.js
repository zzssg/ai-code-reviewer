import { describe, it, expect } from "vitest";
import { chunkJavaByMethods } from "./repo-indexer.utils.js";
import fs from "fs";

describe("chunkJavaByMethods", () => {
    it("JAVA: constructor detection", () => {
    const src = [
      "public class A {",
      "  public A() {",
      "    System.out.println(\"init\");",
      "  }",
      "}"
    ].join("\n");

    const chunks = chunkJavaByMethods(src);
    expect(chunks.length).toBe(1);
    expect(chunks[0].function_name).toBe("A");
  });

  it("JAVA: single simple method", () => {
    const src = [
      "public class A {",
      "  public void foo() {",
      "    int x = 1;",
      "  }",
      "}"
    ].join("\n");

    const chunks = chunkJavaByMethods(src);
    expect(chunks.length).toBe(1);
    expect(chunks[0].function_name).toBe("foo");
    expect(chunks[0].start_line).toBe(2);
    expect(chunks[0].end_line).toBe(4);
  });

  it("JAVA: two simple methods one-liners", () => {
    const src = [
      "public class A {",
      "  public void foo() { return null;}",
      "  public void bar() { return null; }",
      "}"
    ].join("\n");

    const chunks = chunkJavaByMethods(src);
    expect(chunks.length).toBe(2);
    expect(chunks[0].function_name).toBe("foo");
    expect(chunks[1].function_name).toBe("bar");
  });

  it("JAVA: two simple methods", () => {
    const javaSrc = [
      "package com.example;",
      "public class A {",
      "  public void foo() {",
      "    System.out.println(\"x\");",
      "  }",
      "  private int bar(int x) {",
      "    return x * 2;",
      "  }",
      "}"
    ].join("\r\n");

    const chunks = chunkJavaByMethods(javaSrc);
    expect(chunks.length).toBe(2);
    expect(chunks[0].function_name).toBe("foo");
    expect(chunks[0].start_line).toBe(3);
    expect(chunks[0].end_line).toBe(5);
    expect(chunks[1].function_name).toBe("bar");
    expect(chunks[1].start_line).toBe(6);
  });

  it("JAVA: 2 methods throwing Exceptions into 2 chunks with correct names and line numbers", () => {
    const javaSrc = [
      "package com.example;",
      "public class A {",
      "  public void foo() throws Exception {",
      "    System.out.println(\"x\");",
      "  }",
      "  private int bar(int x) throws Exception {",
      "    return x * 2;",
      "  }",
      "}"
    ].join("\r\n");

    const chunks = chunkJavaByMethods(javaSrc);
    expect(chunks.length).toBe(2);
    expect(chunks[0].function_name).toBe("foo");
    expect(chunks[0].start_line).toBe(3);
    expect(chunks[0].end_line).toBe(5);
    expect(chunks[1].function_name).toBe("bar");
    expect(chunks[1].start_line).toBe(6);
  });

    it("JAVA: 2 methods into 2 chunks with correct names and line numbers and empty lines between methods dropped", () => {
    const javaSrc = [
      "package com.example;",
      "public class A {",
      "  public void foo() {",
      "    System.out.println(\"x\");",
      "  }",
      "",
      " ",
      "  private int bar(int x) {",
      "    return x * 2;",
      "  }",
      "}"
    ].join("\r\n");

    const chunks = chunkJavaByMethods(javaSrc);
    expect(chunks.length).toBe(2);
    expect(chunks[0].function_name).toBe("foo");
    expect(chunks[0].start_line).toBe(3);
    expect(chunks[0].end_line).toBe(5);
    expect(chunks[1].function_name).toBe("bar");
    expect(chunks[1].start_line).toBe(8);
    expect(chunks[1].end_line).toBe(10);
  });

  it("JAVA: empty class -> full_file chunk", () => {
    const javaSrc = [
      "package com.example;",
      "// no methods here",
      "public class Empty {}"
    ].join("\n");

    const chunks = chunkJavaByMethods(javaSrc);
    expect(chunks.length).toBe(1);
    expect(chunks[0].function_name).toBe("full_file");
    expect(chunks[0].text).toBe(javaSrc);
  });

  it("JAVA: no methods -> full_file chunk", () => {
    const src = [
      "public class A {",
      "  private int x;",
      "}"
    ].join("\n");

    const chunks = chunkJavaByMethods(src);
    expect(chunks.length).toBe(1);
    expect(chunks[0].function_name).toBe("full_file");
  });

  it("JAVA: field declaration should not be method", () => {
    const src = [
      "public class A {",
      "  private int x = compute();",
      "}"
    ].join("\n");

    const chunks = chunkJavaByMethods(src);
    expect(chunks[0].function_name).toBe("full_file");
  });

  it("JAVA: 2 annotated methods into 2 chunks", () => {
    const javaSrc = [
      "package com.example;",
      "public class A {",
      "  public void foo() {",
      "    System.out.println(\"x\");",
      "  }",
      "  @Override",
      "  private int bar(int x) {",
      "    return x * 2;",
      "  }",
      "}"
    ].join("\n");

    const chunks = chunkJavaByMethods(javaSrc);
    expect(chunks.length).toBe(2);
    expect(chunks[0].function_name).toBe("foo");
    expect(chunks[0].start_line).toBe(3);
    expect(chunks[0].end_line).toBe(5);
    expect(chunks[1].function_name).toBe("bar");
    expect(chunks[1].start_line).toBe(6);
    expect(chunks[1].end_line).toBe(9);
  });
it("JAVA: 2 annotated methods into 2 chunks with surrounding spaces", () => {
    const javaSrc = [
      "package com.example;",
      "public class A {",
      "  public void foo() {",
      "    System.out.println(\"x\");",
      "  }",
      "",
      "  @Override",
      "  @GetMapping(\"/bar\")",
      "  private int bar(int x) {",
      "    return x * 2;",
      "  }",
      "}"
    ].join("\n");

    const chunks = chunkJavaByMethods(javaSrc);
    expect(chunks.length).toBe(2);
    expect(chunks[0].function_name).toBe("foo");
    expect(chunks[0].start_line).toBe(3);
    expect(chunks[0].end_line).toBe(5);
    expect(chunks[1].function_name).toBe("bar");
    expect(chunks[1].start_line).toBe(7);
    expect(chunks[1].end_line).toBe(11);
  });

  it("JAVA: 2 annotated methods into 2 chunks with proper comments handling", () => {
    const javaSrc = [
      "package com.example;",
      "public class A {",
      "  public void foo() {",
      "    System.out.println(\"x\");",
      "  }",
      "",
      "// No need to capture one-line comment",
      "  @Override",
      "  @GetMapping(\"/bar\")",
      "  private int bar(int x) {",
      "    return x * 2;",
      "  }",
      "}"
    ].join("\n");

    const chunks = chunkJavaByMethods(javaSrc);
    expect(chunks.length).toBe(2);
    expect(chunks[0].function_name).toBe("foo");
    expect(chunks[0].start_line).toBe(3);
    expect(chunks[0].end_line).toBe(5);
    expect(chunks[1].function_name).toBe("bar");
    expect(chunks[1].start_line).toBe(8);
    expect(chunks[1].end_line).toBe(12);
  });

    it("JAVA: 2 annotated methods into 2 chunks with proper comments handling 2", () => {
    const javaSrc = [
      "package com.example;",
      "public class A {",
      "  public void foo() {",
      "    System.out.println(\"x\");",
      "  }",
      "",
      "/* No need to capture multi-line comments",
      " * line 2 of comment",
      " */",
      "  @Override",
      "  @GetMapping(\"/bar\")",
      "  private int bar(int x) {",
      "    return x * 2;",
      "  }",
      "}"
    ].join("\n");

    const chunks = chunkJavaByMethods(javaSrc);
    expect(chunks.length).toBe(2);
    expect(chunks[0].function_name).toBe("foo");
    expect(chunks[0].start_line).toBe(3);
    expect(chunks[0].end_line).toBe(5);
    expect(chunks[1].function_name).toBe("bar");
    expect(chunks[1].start_line).toBe(10);
    expect(chunks[1].end_line).toBe(14);
  });

  it("JAVA: parses AllocationBenchmark.java and finds main benchmark methods and helpers", async () => {
    const content = await fs.promises.readFile(new URL("./AllocationBenchmark.java", import.meta.url), "utf8");
    const chunks = chunkJavaByMethods(content);
    const names = chunks.map(c => c.function_name);
    console.log("Extracted method names:", names);
    expect(names.length).toBe(5);
    expect(names).toContain("setUp");
    expect(names).toContain("toInt");
    expect(names).toContain("measureExclusionOnZoneAwareStartedShard");
    expect(names).toContain("measureShardRelocationComplete");
    expect(names).toContain("setUpClusterNodes");
  });

  it("JAVA: parses AvailableIndexFoldersBenchmark.java and finds methods", async () => {
    const content = await fs.promises.readFile(new URL("./AvailableIndexFoldersBenchmark.java", import.meta.url), "utf8");
    const chunks = chunkJavaByMethods(content);
    const names = chunks.map(c => c.function_name);
    console.log("Extracted method names:", names);
    expect(names.length).toBe(3);
    expect(names).toContain("setup");
    expect(names).toContain("availableIndexFolderNaive");
    expect(names).toContain("availableIndexFolderOptimized");
  });

  it("JAVA: parses RoundableSupplier.java and finds methods", async () => {
    const content = await fs.promises.readFile(new URL("./RoundableSupplier.java", import.meta.url), "utf8");
    const chunks = chunkJavaByMethods(content);
    const names = chunks.map(c => c.function_name);
    console.log("Extracted method names:", names);
    expect(names.length).toBe(2);
    expect(names).toContain("RoundableSupplier");
    expect(names).toContain("get");
  });

  it("JAVA: static initializer block", () => {
    const src = [
      "public class A {",
      "  static {",
      "    System.out.println(\"boot\");",
      "  }",
      "}"
    ].join("\n");

    const chunks = chunkJavaByMethods(src);
    expect(chunks.length).toBe(1);
    expect(chunks[0].function_name).toBe("static_initializer");
  });

  it("JAVA: method with dotted return type", () => {
    const src = [
      "public class A {",
      "  private DiscoveryNodes.Builder setUpClusterNodes() {",
      "    return null;",
      "  }",
      "}"
    ].join("\n");

    const chunks = chunkJavaByMethods(src);
    expect(chunks[0].function_name).toBe("setUpClusterNodes");
  });

  it("JAVA: method with generics return type", () => {
    const src = [
      "public class A {",
      "  public Map<String, List<Foo.Bar>> build() {",
      "    return null;",
      "  }",
      "}"
    ].join("\n");

    const chunks = chunkJavaByMethods(src);
    expect(chunks[0].function_name).toBe("build");
  });

  it("JAVA: ignore commented-out method (line comment)", () => {
    const src = [
      "public class A {",
      "  // public void fake() { return null; }",
      "  public void real() { return null; }",
      "}"
    ].join("\n");

    const chunks = chunkJavaByMethods(src);
    expect(chunks.length).toBe(1);
    expect(chunks[0].function_name).toBe("real");
  });

  it("JAVA: ignore commented-out method (block comment)", () => {
    const src = [
      "public class A {",
      "  /*",
      "    public void fake() {}",
      "  */",
      "  public void real() {}",
      "}"
    ].join("\n");

    const chunks = chunkJavaByMethods(src);
    expect(chunks.length).toBe(1);
    expect(chunks[0].function_name).toBe("real");
  });

  it("JAVA: ignore control flow if()", () => {
    const src = [
      "public class A {",
      "  public void foo() {",
      "    if (true) {",
      "      System.out.println(\"x\");",
      "    }",
      "  }",
      "}"
    ].join("\n");

    const chunks = chunkJavaByMethods(src);
    expect(chunks.length).toBe(1);
  });

  it("JAVA: ignore lambda expressions", () => {
    const src = [
      "public class A {",
      "  public void foo() {",
      "    Runnable r = () -> {",
      "      System.out.println(\"x\");",
      "    };",
      "  }",
      "}"
    ].join("\n");

    const chunks = chunkJavaByMethods(src);
    expect(chunks.length).toBe(1);
  });

  it("JAVA: include annotations above method", () => {
    const src = [
      "public class A {",
      "  @Override",
      "  @Deprecated",
      "  public void foo() {}",
      "}"
    ].join("\n");

    const chunks = chunkJavaByMethods(src);
    expect(chunks[0].start_line).toBe(2);
  });

  it("JAVA: exclude empty lines around method", () => {
    const src = [
      "public class A {",
      "",
      "  public void foo() {",
      "  }",
      "",
      "}"
    ].join("\n");

    const chunks = chunkJavaByMethods(src);
    expect(chunks[0].start_line).toBe(3);
    expect(chunks[0].end_line).toBe(4);
  });

  it("JAVA: nested braces inside method", () => {
    const src = [
      "public class A {",
      "  public void foo() {",
      "    for(int i=0;i<10;i++){",
      "      if(i>5){",
      "        System.out.println(i);",
      "      }",
      "    }",
      "  }",
      "}"
    ].join("\n");

    const chunks = chunkJavaByMethods(src);
    expect(chunks.length).toBe(1);
  });

  it("JAVA: method with throws clause", () => {
    const src = [
      "public class A {",
      "  public void foo() throws Exception {",
      "  }",
      "}"
    ].join("\n");

    const chunks = chunkJavaByMethods(src);
    expect(chunks[0].function_name).toBe("foo");
  });

  it("JAVA: multiple methods and constructor", () => {
    const src = [
      "public class A {",
      "  public A() {}",
      "  public void foo() {}",
      "  private void bar() {}",
      "}"
    ].join("\n");

    const chunks = chunkJavaByMethods(src);
    expect(chunks.length).toBe(3);
  });

  it("JAVA: interface default method", () => {
    const src = [
      "public interface A {",
      "  default void foo() {",
      "  }",
      "}"
    ].join("\n");

    const chunks = chunkJavaByMethods(src);
    expect(chunks[0].function_name).toBe("foo");
  });

  it("JAVA: abstract method without body should not chunk", () => {
    const src = [
      "public abstract class A {",
      "  public abstract void foo();",
      "}"
    ].join("\n");

    const chunks = chunkJavaByMethods(src);
    expect(chunks[0].function_name).toBe("full_file");
  });

  it("JAVA: split class with 2 annotated methods into 2 chunks with proper comments handling", () => {
    const javaSrc = [
      "package com.example;",
      "public class A {",
      "  public void foo() {",
      "    System.out.println(\"x\");",
      "  }",
      "",
      "/* No need to capture multi-line comments",
      " * line 2 of comment",
      " */",
      "  @Override",
      "  @GetMapping(\"/bar\")",
      "  private int bar(int x) {",
      "    return x * 2;",
      "  }",
      "}"
    ].join("\n");

    const chunks = chunkJavaByMethods(javaSrc);

    expect(chunks.length).toBe(2);
    expect(chunks[0].function_name).toBe("foo");
    expect(chunks[0].start_line).toBe(3);
    expect(chunks[0].end_line).toBe(5);
    expect(chunks[1].function_name).toBe("bar");
    expect(chunks[1].start_line).toBe(10);
    expect(chunks[1].end_line).toBe(14);
  });
});