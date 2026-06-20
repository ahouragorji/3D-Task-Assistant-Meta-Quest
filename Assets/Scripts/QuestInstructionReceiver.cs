using System;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class QuestInstructionReceiver : MonoBehaviour
{
    // -------------------------------------------------------------------------
    // Inspector fields
    // -------------------------------------------------------------------------

    [Header("References")]
    public QuestPassthroughSender sender;

    [Header("Arrow Prefabs")]
    [Tooltip("Arrow pointing downward — spawned ABOVE the object (orientation: 'up').")]
    public GameObject arrowUpPrefab;
    [Tooltip("Arrow pointing upward — spawned BELOW the object (orientation: 'down').")]
    public GameObject arrowDownPrefab;
    [Tooltip("Arrow pointing backward — spawned IN FRONT of object (orientation: 'front').")]
    public GameObject arrowFrontPrefab;
    [Tooltip("Arrow pointing forward — spawned BEHIND the object (orientation: 'back').")]
    public GameObject arrowBackPrefab;
    [Tooltip("Arrow pointing rightward — spawned to the LEFT (orientation: 'left').")]
    public GameObject arrowLeftPrefab;
    [Tooltip("Arrow pointing leftward — spawned to the RIGHT (orientation: 'right').")]
    public GameObject arrowRightPrefab;

    [Tooltip("How far from the object's world point the arrow is offset, in metres.")]
    public float arrowOffset = 0.3f;
    [Tooltip("Uniform scale applied to every spawned arrow.")]
    public float arrowScale = 0.15f;

    [Header("Navigation Buttons")]
    public Button nextButton;
    public Button previousButton;

    [Header("Instruction Panel")]
    public TMP_Text instructionText;
    public TMP_Text stepCounterText;

    [Header("Debug — Bounding Boxes")]
    public bool showDebugBoundingBoxes = false;
    public Button debugToggleButton;
    [SerializeField] private Material drawMaterial;

    // -------------------------------------------------------------------------
    // Private state
    // -------------------------------------------------------------------------

    private readonly Dictionary<string, string[]> _chunkBuffers  = new Dictionary<string, string[]>();
    private readonly Dictionary<int, List<GameObject>> _arrowsByStep    = new Dictionary<int, List<GameObject>>();
    private readonly Dictionary<int, List<GameObject>> _bboxEdgesByStep = new Dictionary<int, List<GameObject>>();

    private InstructionResponse _currentResponse;
    private int _currentStepIndex = 0;
    private int _maxStep          = 0;

    // -------------------------------------------------------------------------
    // Orientation helpers
    // -------------------------------------------------------------------------


    private Vector3 GetCalculatedSpawnPosition(InstructionPlacement placement, string orientation)
{
    // 1. Get the object's center point
    Vector3 center = new Vector3(placement.worldX, placement.worldY, placement.worldZ);

    // 2. Determine the normalized direction vector we want the arrow to spawn in
    Vector3 dir = Vector3.up; 

    // 2. Determine the normalized direction vector we want the arrow to spawn in

    // USE THE SENDER'S SAVED POSITION INSTEAD OF THE LIVE CAMERA
    if (sender != null)
    {
        // 1. Draw a line from Object to the User's SAVED physical position
        Vector3 objectToUser = sender.lastCaptureHeadPosition - center;
        objectToUser.y = 0; // Keep it perfectly flat
        objectToUser.Normalize();

        // 2. The exact opposite (a line from the User to the Object)
        Vector3 userToObject = -objectToUser;

        // 3. Calculate true "Right" and "Left" relative to the user looking at the object
        Vector3 userRight = Vector3.Cross(Vector3.up, userToObject).normalized;

        switch (orientation)
        {
            case "up":    dir = Vector3.up; break;
            case "down":  dir = Vector3.down; break;
            case "front": dir = objectToUser; break; 
            case "back":  dir = userToObject; break;  
            case "left":  dir = -userRight; break;
            case "right": dir = userRight; break;
        }
    }


    // 3. Find how far the bounding box extends in that specific direction
    float edgeDistance = 0f;

    if (placement.bboxCorners != null && placement.bboxCorners.Length == 8)
    {
        foreach (var corner in placement.bboxCorners)
        {
            // Vector pointing from the center of the object to this corner
            Vector3 centerToCorner = corner.ToVector3() - center;
            
            // Project that vector onto our chosen direction using a Dot Product.
            // This tells us how far this corner "sticks out" in our target direction.
            float projection = Vector3.Dot(centerToCorner, dir);
            
            if (projection > edgeDistance)
            {
                edgeDistance = projection; // Save the furthest extent
            }
        }
    }

    // 4. Calculate final position: Center + Edge of Box + Configurable Padding
    return center + (dir * (edgeDistance + arrowOffset));
}

// private Vector3 OffsetForOrientation(string orientation)
// {
//     // Ensure we have a camera reference (the Quest headset)
//     if (Camera.main == null) return Vector3.up * arrowOffset;

//     Transform head = Camera.main.transform;

//     // Get the direction the user is currently looking, flattened so arrows don't drift vertically
//     Vector3 camForward = head.forward;
//     camForward.y = 0;
//     camForward.Normalize();

//     Vector3 camRight = head.right;
//     camRight.y = 0;
//     camRight.Normalize();

//     switch (orientation)
//     {
//         case "up":    return Vector3.up * arrowOffset;
//         case "down":  return Vector3.down * arrowOffset;
        
//         case "front": return -camForward * arrowOffset;
        
//         case "back":  return camForward * arrowOffset;
        
//         case "left":  return -camRight * arrowOffset;
//         case "right": return camRight * arrowOffset;
        
//         default:      return Vector3.up * arrowOffset;
//     }
// }

    private GameObject PrefabForOrientation(string orientation)
    {
        switch (orientation)
        {
            case "up":    return arrowUpPrefab;
            case "down":  return arrowDownPrefab;
            case "front": return arrowFrontPrefab;
            case "back":  return arrowBackPrefab;
            case "left":  return arrowLeftPrefab;
            case "right": return arrowRightPrefab;
            default:      return arrowUpPrefab;
        }
    }

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    private void Awake()
    {
        // Log every prefab slot so we know immediately which are unassigned.
        Debug.Log($"[QIR:Awake] Prefab slots — " +
                  $"up={arrowUpPrefab?.name ?? "NULL"}, " +
                  $"down={arrowDownPrefab?.name ?? "NULL"}, " +
                  $"front={arrowFrontPrefab?.name ?? "NULL"}, " +
                  $"back={arrowBackPrefab?.name ?? "NULL"}, " +
                  $"left={arrowLeftPrefab?.name ?? "NULL"}, " +
                  $"right={arrowRightPrefab?.name ?? "NULL"}");

        if (drawMaterial == null)
        {
            Shader fallback = Shader.Find("Sprites/Default");
            if (fallback != null)
                drawMaterial = new Material(fallback);
            else
                Debug.LogWarning("[QIR:Awake] drawMaterial null and Sprites/Default not found.");
        }
    }

    private void OnEnable()
    {
        if (sender != null)
            sender.OnAppMessageReceived += HandleAppMessage;
        else
            Debug.LogError("[QIR:OnEnable] 'sender' not assigned in Inspector.");

        if (nextButton        != null) nextButton.onClick.AddListener(AdvanceToNextStep);
        if (previousButton    != null) previousButton.onClick.AddListener(ReturnToPreviousStep);
        if (debugToggleButton != null) debugToggleButton.onClick.AddListener(ToggleDebugBoundingBoxes);
    }

    private void OnDisable()
    {
        if (sender            != null) sender.OnAppMessageReceived -= HandleAppMessage;
        if (nextButton        != null) nextButton.onClick.RemoveListener(AdvanceToNextStep);
        if (previousButton    != null) previousButton.onClick.RemoveListener(ReturnToPreviousStep);
        if (debugToggleButton != null) debugToggleButton.onClick.RemoveListener(ToggleDebugBoundingBoxes);
    }

    // -------------------------------------------------------------------------
    // Message handling
    // -------------------------------------------------------------------------

    private void HandleAppMessage(string message)
    {
        if (!message.StartsWith("INSTR|")) return;

        string[] parts = message.Split(new char[] { '|' }, 5);
        if (parts.Length < 5)
        {
            Debug.LogError($"[QIR:Chunk] Malformed: expected 5 parts, got {parts.Length}");
            return;
        }

        string id    = parts[1];
        int    index = int.Parse(parts[2]);
        int    total = int.Parse(parts[3]);
        string data  = parts[4];

        if (!_chunkBuffers.ContainsKey(id))
        {
            _chunkBuffers[id] = new string[total];
            Debug.Log($"[QIR:Chunk] Started '{id}': expecting {total} chunks.");
        }

        _chunkBuffers[id][index] = data;
        int received = 0;
        foreach (var c in _chunkBuffers[id]) if (c != null) received++;
        Debug.Log($"[QIR:Chunk] '{id}': {received}/{total} chunks received.");

        if (Array.TrueForAll(_chunkBuffers[id], c => c != null))
        {
            string fullBase64 = string.Concat(_chunkBuffers[id]);
            _chunkBuffers.Remove(id);
            Debug.Log($"[QIR:Chunk] '{id}' complete — {fullBase64.Length} base64 chars. Decoding.");
            DecodeAndApply(fullBase64);
        }
    }

    private void DecodeAndApply(string base64Json)
    {
        string json = null;
        InstructionResponse parsed = null;

        // ── 1. Base64 decode ──
        try
        {
            byte[] bytes = Convert.FromBase64String(base64Json);
            json = System.Text.Encoding.UTF8.GetString(bytes);
            Debug.Log($"[QIR:Decode] JSON ({json.Length} chars): {json.Substring(0, Mathf.Min(300, json.Length))}...");
        }
        catch (Exception e)
        {
            Debug.LogError($"[QIR:Decode] Base64 decode failed: {e.Message}");
            return;
        }

        // ── 2. JSON parse ──
        try
        {
            parsed = JsonUtility.FromJson<InstructionResponse>(json);
        }
        catch (Exception e)
        {
            Debug.LogError($"[QIR:Decode] JSON parse failed: {e.Message}");
            return;
        }

        if (parsed == null)
        {
            Debug.LogError("[QIR:Decode] JsonUtility returned null — JSON structure may not match InstructionResponse.");
            return;
        }

        if (parsed.placements == null)
        {
            Debug.LogError("[QIR:Decode] parsed.placements is null — 'placements' key missing or misspelled in JSON.");
            return;
        }

        Debug.Log($"[QIR:Decode] Parsed OK — id='{parsed.id}', placements={parsed.placements.Length}");

        if (parsed.placements.Length == 0)
        {
            Debug.LogWarning("[QIR:Decode] Zero placements in payload — nothing to spawn.");
            return;
        }

        // ── 3. Log every placement so we can verify the data ──
        for (int i = 0; i < parsed.placements.Length; i++)
        {
            var p = parsed.placements[i];
            Debug.Log($"[QIR:Placement {i}] step={p.step} label='{p.label}' orientation='{p.orientation}' " +
                      $"world=({p.worldX:F3},{p.worldY:F3},{p.worldZ:F3}) " +
                      $"corners={p.bboxCorners?.Length ?? -1}");
        }

        // ── 4. Normalize step numbers to a clean 1-based contiguous sequence ──
        // The server emits one placement per detected object, so a 3-step task
        // with 2 objects in step 1 produces step values like [1,1,2,3] which is
        // fine — but if any step produced only text-only placements the numbers
        // can jump (e.g. [1,3,3]) and _currentStepIndex+1 never matches step 2,
        // so RefreshVisibility activates the wrong set of arrows.
        // Collect the unique step values in sorted order and remap them.
        var uniqueSteps = new List<int>();
        foreach (var p in parsed.placements)
            if (!uniqueSteps.Contains(p.step)) uniqueSteps.Add(p.step);
        uniqueSteps.Sort();

        var stepRemap = new Dictionary<int, int>();
        for (int i = 0; i < uniqueSteps.Count; i++)
            stepRemap[uniqueSteps[i]] = i + 1;  // remap to 1-based

        bool remapNeeded = false;
        foreach (var p in parsed.placements)
            if (stepRemap[p.step] != p.step) { remapNeeded = true; break; }

        if (remapNeeded)
        {
            Debug.Log($"[QIR:Decode] Remapping steps: {string.Join(", ", uniqueSteps)} → 1..{uniqueSteps.Count}");
            foreach (var p in parsed.placements)
                p.step = stepRemap[p.step];
        }

        _currentResponse  = parsed;
        _currentStepIndex = 0;

        _maxStep = 0;
        foreach (var p in _currentResponse.placements)
            if (p.step > _maxStep) _maxStep = p.step;

        Debug.Log($"[QIR:Decode] maxStep={_maxStep}. Calling SpawnAllArrows.");
        SpawnAllArrows();
        Debug.Log($"[QIR:Decode] Calling RefreshVisibility.");
        RefreshVisibility();
        Debug.Log($"[QIR:Decode] Calling ShowCurrentStepText.");
        ShowCurrentStepText();
    }

    // -------------------------------------------------------------------------
    // Arrow spawning
    // -------------------------------------------------------------------------

    private void SpawnAllArrows()
    {
        ClearAllSpawnedObjects();

        int spawnedCount  = 0;
        int skippedNoLabel = 0;
        int skippedNoPrefab = 0;

        foreach (var placement in _currentResponse.placements)
        {
            int stepNum = placement.step;

            if (!_arrowsByStep.ContainsKey(stepNum))    _arrowsByStep[stepNum]    = new List<GameObject>();
            if (!_bboxEdgesByStep.ContainsKey(stepNum)) _bboxEdgesByStep[stepNum] = new List<GameObject>();

            // Text-only placement — instruction with no detected object.
            if (string.IsNullOrEmpty(placement.label))
            {
                skippedNoLabel++;
                Debug.Log($"[QIR:Spawn] Step {stepNum}: text-only placement, no arrow.");
                continue;
            }

            string orientation = string.IsNullOrEmpty(placement.orientation) ? "up" : placement.orientation;
            GameObject prefab  = PrefabForOrientation(orientation);

            // ── Critical prefab null check — stop here if unassigned ──
            if (prefab == null)
            {
                skippedNoPrefab++;
                Debug.LogError($"[QIR:Spawn] Step {stepNum} label='{placement.label}' orientation='{orientation}': " +
                               $"prefab is NULL. Assign '{orientation}ArrowPrefab' in the Inspector on " +
                               $"the QuestInstructionReceiver component. Skipping this placement.");
                continue;   // ← was missing before; code fell through to Instantiate(null)
            }

            // Vector3 objectPos = new Vector3(placement.worldX, placement.worldY, placement.worldZ);
            // Vector3 offset    = OffsetForOrientation(orientation);
            // Vector3 spawnPos  = objectPos + offset;

            Vector3 objectPos = new Vector3(placement.worldX, placement.worldY, placement.worldZ);
            Vector3 spawnPos  = GetCalculatedSpawnPosition(placement, orientation);

            Quaternion spawnRot = (spawnPos != objectPos)
                ? Quaternion.LookRotation(objectPos - spawnPos, Vector3.up)
                : Quaternion.identity;

            Debug.Log($"[QIR:Spawn] Step {stepNum} '{placement.label}' [{orientation}] — " +
            $"objectPos={objectPos}, spawnPos={spawnPos}, prefab={prefab.name}");

            GameObject arrow = Instantiate(prefab, spawnPos, spawnRot);
            arrow.transform.localScale = Vector3.one * arrowScale;
            arrow.name = $"Arrow_{orientation}_Step{stepNum}_{placement.label}";

            // Intentionally spawned ACTIVE. RefreshVisibility (called right after
            // this method) will deactivate steps that aren't the current step.
            // Never SetActive(false) here — children added to an inactive parent
            // cannot be re-activated with SetActive(true) on the child alone.
            _arrowsByStep[stepNum].Add(arrow);
            spawnedCount++;

            Debug.Log($"[QIR:Spawn] Instantiated '{arrow.name}' at {spawnPos}. " +
                      $"activeSelf={arrow.activeSelf} activeInHierarchy={arrow.activeInHierarchy}");

            // Debug bounding box edges.
            if (placement.bboxCorners != null && placement.bboxCorners.Length == 8)
            {
                List<GameObject> edges = SpawnBoundingBoxEdges(arrow, placement.bboxCorners);
                _bboxEdgesByStep[stepNum].AddRange(edges);
            }
            else
            {
                Debug.Log($"[QIR:Spawn] No bbox corners for '{placement.label}' " +
                          $"(corners={(placement.bboxCorners == null ? "null" : placement.bboxCorners.Length.ToString())}).");
            }
        }

        Debug.Log($"[QIR:Spawn] Done — spawned={spawnedCount}, skipped(no label)={skippedNoLabel}, " +
                  $"skipped(no prefab)={skippedNoPrefab}, steps with arrows={_arrowsByStep.Count}.");
    }

    private void ClearAllSpawnedObjects()
    {
        int destroyed = 0;
        foreach (var list in _arrowsByStep.Values)
            foreach (var go in list)
                if (go != null) { Destroy(go); destroyed++; }

        _arrowsByStep.Clear();
        _bboxEdgesByStep.Clear();
        if (destroyed > 0) Debug.Log($"[QIR:Clear] Destroyed {destroyed} previous arrows.");
    }

    // -------------------------------------------------------------------------
    // Visibility
    // -------------------------------------------------------------------------

    private void RefreshVisibility()
    {
        int activeStep = _currentStepIndex + 1;
        Debug.Log($"[QIR:Visibility] activeStep={activeStep}, _arrowsByStep keys=[{string.Join(",", _arrowsByStep.Keys)}]");

        foreach (var kvp in _arrowsByStep)
        {
            bool active = (kvp.Key == activeStep);
            foreach (var arrow in kvp.Value)
            {
                if (arrow == null) continue;
                arrow.SetActive(active);
                Debug.Log($"[QIR:Visibility] '{arrow.name}' SetActive({active}) → " +
                          $"activeSelf={arrow.activeSelf} activeInHierarchy={arrow.activeInHierarchy}");
            }
        }

        foreach (var kvp in _bboxEdgesByStep)
        {
            bool edgeActive = (kvp.Key == activeStep) && showDebugBoundingBoxes;
            foreach (var edge in kvp.Value)
                if (edge != null) edge.SetActive(edgeActive);
        }

        RefreshNavButtonState();
    }

    private void RefreshNavButtonState()
    {
        if (previousButton != null) previousButton.interactable = (_currentStepIndex > 0);
        if (nextButton     != null) nextButton.interactable     = (_currentStepIndex < _maxStep - 1);
    }

    // -------------------------------------------------------------------------
    // Instruction text
    // -------------------------------------------------------------------------

    private void ShowCurrentStepText()
    {
        if (_currentResponse == null) return;
        int activeStep = _currentStepIndex + 1;
        string text = null;
        foreach (var p in _currentResponse.placements)
            if (p.step == activeStep) { text = p.instruction; break; }

        if (instructionText != null) instructionText.text = text ?? "(no instruction for this step)";
        if (stepCounterText != null) stepCounterText.text = $"Step {activeStep} / {_maxStep}";
    }

    // -------------------------------------------------------------------------
    // Navigation
    // -------------------------------------------------------------------------

    public void AdvanceToNextStep()
    {
        if (_currentResponse == null || _currentStepIndex >= _maxStep - 1) return;
        _currentStepIndex++;
        RefreshVisibility();
        ShowCurrentStepText();
    }

    public void ReturnToPreviousStep()
    {
        if (_currentResponse == null || _currentStepIndex <= 0) return;
        _currentStepIndex--;
        RefreshVisibility();
        ShowCurrentStepText();
    }

    public void ToggleDebugBoundingBoxes()
    {
        showDebugBoundingBoxes = !showDebugBoundingBoxes;
        int activeStep = _currentStepIndex + 1;
        if (_bboxEdgesByStep.TryGetValue(activeStep, out var edges))
            foreach (var edge in edges)
                if (edge != null) edge.SetActive(showDebugBoundingBoxes);
        Debug.Log($"[QIR] Debug bounding boxes: {(showDebugBoundingBoxes ? "ON" : "OFF")}");
    }

    // -------------------------------------------------------------------------
    // Debug bounding box edges
    // -------------------------------------------------------------------------

    private List<GameObject> SpawnBoundingBoxEdges(GameObject parent, Corner3D[] corners)
    {
        var edges = new List<GameObject>();
        if (corners == null || corners.Length != 8) return edges;

        Vector3[] pts = new Vector3[8];
        for (int i = 0; i < 8; i++) pts[i] = corners[i].ToVector3();

        Color c = Color.cyan;
        edges.Add(CreateEdge(parent, pts[0], pts[1], c)); edges.Add(CreateEdge(parent, pts[1], pts[2], c));
        edges.Add(CreateEdge(parent, pts[2], pts[3], c)); edges.Add(CreateEdge(parent, pts[3], pts[0], c));
        edges.Add(CreateEdge(parent, pts[4], pts[5], c)); edges.Add(CreateEdge(parent, pts[5], pts[6], c));
        edges.Add(CreateEdge(parent, pts[6], pts[7], c)); edges.Add(CreateEdge(parent, pts[7], pts[4], c));
        edges.Add(CreateEdge(parent, pts[0], pts[4], c)); edges.Add(CreateEdge(parent, pts[1], pts[5], c));
        edges.Add(CreateEdge(parent, pts[2], pts[6], c)); edges.Add(CreateEdge(parent, pts[3], pts[7], c));

        // Edges start hidden; RefreshVisibility enables them when the step is
        // active AND showDebugBoundingBoxes is true. They are parented to an
        // ACTIVE arrow so SetActive works correctly.
        foreach (var e in edges) e.SetActive(false);
        return edges;
    }

    private GameObject CreateEdge(GameObject parent, Vector3 start, Vector3 end, Color color)
    {
        var obj = new GameObject("BBoxEdge");
        obj.transform.SetParent(parent.transform);
        var lr = obj.AddComponent<LineRenderer>();
        lr.positionCount = 2;
        lr.SetPosition(0, start);
        lr.SetPosition(1, end);
        lr.startWidth    = 0.005f;
        lr.endWidth      = 0.005f;
        lr.material      = drawMaterial;
        lr.startColor    = color;
        lr.endColor      = color;
        lr.useWorldSpace = true;
        return obj;
    }

    // -------------------------------------------------------------------------
    // Serialization types
    // -------------------------------------------------------------------------

    [Serializable]
    private class InstructionPlacement
    {
        public int        step;
        public string     instruction;
        public string     label;
        public string     orientation;
        public float      worldX, worldY, worldZ;
        public Corner3D[] bboxCorners;
    }

    [Serializable]
    public class Corner3D
    {
        public float x, y, z;
        public Vector3 ToVector3() => new Vector3(x, y, z);
    }

    [Serializable]
    private class InstructionResponse
    {
        public string               id;
        public InstructionPlacement[] placements;
    }
}