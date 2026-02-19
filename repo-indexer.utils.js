export function chunkJavaByMethods(source) {
  const lines = source.split('\n');
  const chunks = [];

  let chunkId = 1;
  let braceDepth = 0;
  let inChunk = false;
  let chunkStart = 0;
  let chunkName = '';
  let className = null;

  let inBlockComment = false;
  let inString = false;

  // ---------------------------------------------------
  // Cleaning & Brace Tracking (comment-aware)
  // ---------------------------------------------------

  function stripComments(line) {
    let result = '';
    let i = 0;

    while (i < line.length) {
      const char = line[i];
      const next = line[i + 1];

      if (!inString && char === '/' && next === '*') {
        inBlockComment = true;
        i += 2;
        continue;
      }

      if (inBlockComment && char === '*' && next === '/') {
        inBlockComment = false;
        i += 2;
        continue;
      }

      if (inBlockComment) {
        i++;
        continue;
      }

      if (!inString && char === '/' && next === '/') {
        break; // ignore rest of line
      }

      if (char === '"' && line[i - 1] !== '\\') {
        inString = !inString;
      }

      if (!inString) result += char;

      i++;
    }

    return result.trim();
  }

  function countBraces(line) {
    let open = 0;
    let close = 0;
    let i = 0;

    while (i < line.length) {
      const char = line[i];
      const next = line[i + 1];

      if (!inString && char === '/' && next === '*') {
        inBlockComment = true;
        i += 2;
        continue;
      }

      if (inBlockComment && char === '*' && next === '/') {
        inBlockComment = false;
        i += 2;
        continue;
      }

      if (inBlockComment) {
        i++;
        continue;
      }

      if (!inString && char === '/' && next === '/') {
        break;
      }

      if (char === '"' && line[i - 1] !== '\\') {
        inString = !inString;
      }

      if (!inString) {
        if (char === '{') open++;
        if (char === '}') close++;
      }

      i++;
    }

    return { open, close };
  }

  // ---------------------------------------------------
  // Detection
  // ---------------------------------------------------

  function detectClass(line) {
    const match = line.match(/\bclass\s+([A-Za-z0-9_]+)/);
    if (match) className = match[1];
  }

  function isAnnotation(line) {
    return line.startsWith('@');
  }

  function isStaticInitializer(line) {
    return /^\s*static\s*\{/.test(line);
  }

  function isConstructor(line) {
    if (!className) return false;
    return new RegExp(`\\b${className}\\s*\\(`).test(line);
  }

  function isControlFlow(line) {
    return /^(if|for|while|switch|catch|do)\s*\(/.test(line);
  }

  function isLambda(line) {
    return line.includes('->');
  }

  function isFieldDeclaration(line) {
    return /^[\w\<\>\[\]\.,\s]+\s+[A-Za-z0-9_]+\s*(=|;)/.test(line);
  }

  // JS-compatible regex (no /x flag)
  function isMethodSignature(line) {
    if (isControlFlow(line)) return false;
    if (isLambda(line)) return false;
    if (isFieldDeclaration(line)) return false;

    return /^\s*(public|protected|private)?\s*(static\s+|final\s+|synchronized\s+|abstract\s+|native\s+|strictfp\s+)*[A-Za-z0-9_<>\[\]\.,\s]+\s+[A-Za-z0-9_]+\s*\(/.test(line);
  }

  function extractMethodName(line) {
    const cleaned = line
      .replace(/<[^>]+>/g, '')
      .replace(/\s+/g, ' ');
    const match = cleaned.match(/([A-Za-z0-9_]+)\s*\(/);
    return match ? match[1] : 'unknown';
  }

  function trimLeadingEmpty(start) {
    while (start < lines.length && lines[start].trim() === '') {
      start++;
    }
    return start;
  }

  function trimTrailingEmpty(end) {
    while (end >= 0 && lines[end].trim() === '') {
      end--;
    }
    return end;
  }

  // ---------------------------------------------------
  // Main Loop
  // ---------------------------------------------------

  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i];
    const clean = stripComments(raw);
    detectClass(clean);

    const { open, close } = countBraces(raw);

    if (!inChunk) {
      const staticBlock = isStaticInitializer(clean);
      const ctor = isConstructor(clean);
      const method = isMethodSignature(clean);

      if (staticBlock || ctor || method) {

        let start = i;

        // include annotations above
        while (start > 0) {
          const prev = stripComments(lines[start - 1]);
          if (isAnnotation(prev)) start--;
          else break;
        }

        start = trimLeadingEmpty(start);

        chunkStart = start;
        braceDepth = open - close;

        if (staticBlock) chunkName = "static_initializer";
        else if (ctor) chunkName = className;
        else chunkName = extractMethodName(clean);

        if (braceDepth === 0 && open > 0) {
          const end = trimTrailingEmpty(i);

          chunks.push({
            chunk_id: chunkId++,
            function_name: chunkName,
            start_line: chunkStart + 1,
            end_line: end + 1,
            text: lines.slice(chunkStart, end + 1).join("\n")
          });

          chunkName = '';
          inChunk = false;
        } else {
          inChunk = true;
        }
      }
    } else {
      braceDepth += open - close;

      if (braceDepth === 0) {
        const end = trimTrailingEmpty(i);

        chunks.push({
          chunk_id: chunkId++,
          function_name: chunkName,
          start_line: chunkStart + 1,
          end_line: end + 1,
          text: lines.slice(chunkStart, end + 1).join('\n')
        });

        inChunk = false;
        chunkName = '';
      }
    }
  }

  // Fallback
  if (chunks.length === 0) {
    return [{
      chunk_id: 1,
      function_name: "full_file",
      start_line: 1,
      end_line: lines.length,
      text: source
    }];
  }

  return chunks;
}
