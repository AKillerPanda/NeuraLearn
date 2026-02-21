import type { Node, Edge } from 'reactflow';

export interface LearningPath {
  id: string;
  name: string;
  description: string;
  duration: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  nodeIds: string[];
}

export interface SkillGraphData {
  nodes: Node[];
  edges: Edge[];
  paths: LearningPath[];
}

// AI-powered knowledge graph generator (mock data for now)
export function generateKnowledgeGraph(skill: string): SkillGraphData {
  const skillLower = skill.toLowerCase();
  
  // Define skill templates for common topics
  const templates: Record<string, SkillGraphData> = {
    'machine learning': generateMLGraph(),
    'react development': generateReactGraph(),
    'data structures': generateDataStructuresGraph(),
    'web development': generateWebDevGraph(),
    'spanish language': generateSpanishGraph(),
    'photography': generatePhotographyGraph(),
  };

  // Check if we have a template
  for (const [key, data] of Object.entries(templates)) {
    if (skillLower.includes(key) || key.includes(skillLower)) {
      return data;
    }
  }

  // Generate generic graph if no template matches
  return generateGenericGraph(skill);
}

function generateMLGraph(): SkillGraphData {
  const nodes: Node[] = [
    {
      id: '1',
      type: 'input',
      data: { label: 'Machine Learning Fundamentals' },
      position: { x: 400, y: 0 },
    },
    {
      id: '2',
      data: { label: 'Python Programming' },
      position: { x: 100, y: 100 },
    },
    {
      id: '3',
      data: { label: 'Linear Algebra' },
      position: { x: 400, y: 100 },
    },
    {
      id: '4',
      data: { label: 'Statistics & Probability' },
      position: { x: 700, y: 100 },
    },
    {
      id: '5',
      data: { label: 'Data Preprocessing' },
      position: { x: 250, y: 220 },
    },
    {
      id: '6',
      data: { label: 'Supervised Learning' },
      position: { x: 550, y: 220 },
    },
    {
      id: '7',
      data: { label: 'Linear Regression' },
      position: { x: 150, y: 340 },
    },
    {
      id: '8',
      data: { label: 'Logistic Regression' },
      position: { x: 350, y: 340 },
    },
    {
      id: '9',
      data: { label: 'Decision Trees' },
      position: { x: 550, y: 340 },
    },
    {
      id: '10',
      data: { label: 'Neural Networks' },
      position: { x: 750, y: 340 },
    },
    {
      id: '11',
      data: { label: 'Model Evaluation' },
      position: { x: 250, y: 460 },
    },
    {
      id: '12',
      type: 'output',
      data: { label: 'Deep Learning' },
      position: { x: 550, y: 460 },
    },
  ];

  const edges: Edge[] = [
    { id: 'e1-2', source: '1', target: '2' },
    { id: 'e1-3', source: '1', target: '3' },
    { id: 'e1-4', source: '1', target: '4' },
    { id: 'e2-5', source: '2', target: '5' },
    { id: 'e3-6', source: '3', target: '6' },
    { id: 'e4-6', source: '4', target: '6' },
    { id: 'e5-7', source: '5', target: '7' },
    { id: 'e6-8', source: '6', target: '8' },
    { id: 'e6-9', source: '6', target: '9' },
    { id: 'e6-10', source: '6', target: '10' },
    { id: 'e7-11', source: '7', target: '11' },
    { id: 'e8-11', source: '8', target: '11' },
    { id: 'e9-12', source: '9', target: '12' },
    { id: 'e10-12', source: '10', target: '12' },
    { id: 'e11-12', source: '11', target: '12' },
  ];

  const paths: LearningPath[] = [
    {
      id: 'path1',
      name: 'Beginner Path',
      description: 'Start with fundamentals and gradually build up to basic ML models',
      duration: '3 months',
      difficulty: 'beginner',
      nodeIds: ['1', '2', '3', '5', '7', '11'],
    },
    {
      id: 'path2',
      name: 'Statistical Approach',
      description: 'Focus on statistical foundations before diving into algorithms',
      duration: '4 months',
      difficulty: 'intermediate',
      nodeIds: ['1', '4', '3', '6', '8', '9', '11', '12'],
    },
    {
      id: 'path3',
      name: 'Fast Track to Deep Learning',
      description: 'Accelerated path for those with programming background',
      duration: '2 months',
      difficulty: 'advanced',
      nodeIds: ['1', '2', '6', '10', '12'],
    },
  ];

  return { nodes, edges, paths };
}

function generateReactGraph(): SkillGraphData {
  const nodes: Node[] = [
    {
      id: '1',
      type: 'input',
      data: { label: 'React Development' },
      position: { x: 400, y: 0 },
    },
    {
      id: '2',
      data: { label: 'JavaScript ES6+' },
      position: { x: 200, y: 100 },
    },
    {
      id: '3',
      data: { label: 'HTML & CSS' },
      position: { x: 600, y: 100 },
    },
    {
      id: '4',
      data: { label: 'JSX Syntax' },
      position: { x: 400, y: 200 },
    },
    {
      id: '5',
      data: { label: 'Components & Props' },
      position: { x: 200, y: 300 },
    },
    {
      id: '6',
      data: { label: 'State & Hooks' },
      position: { x: 600, y: 300 },
    },
    {
      id: '7',
      data: { label: 'Event Handling' },
      position: { x: 100, y: 400 },
    },
    {
      id: '8',
      data: { label: 'useEffect & Lifecycle' },
      position: { x: 350, y: 400 },
    },
    {
      id: '9',
      data: { label: 'React Router' },
      position: { x: 600, y: 400 },
    },
    {
      id: '10',
      data: { label: 'State Management' },
      position: { x: 850, y: 400 },
    },
    {
      id: '11',
      type: 'output',
      data: { label: 'Advanced Patterns' },
      position: { x: 400, y: 500 },
    },
  ];

  const edges: Edge[] = [
    { id: 'e1-2', source: '1', target: '2' },
    { id: 'e1-3', source: '1', target: '3' },
    { id: 'e2-4', source: '2', target: '4' },
    { id: 'e3-4', source: '3', target: '4' },
    { id: 'e4-5', source: '4', target: '5' },
    { id: 'e4-6', source: '4', target: '6' },
    { id: 'e5-7', source: '5', target: '7' },
    { id: 'e6-8', source: '6', target: '8' },
    { id: 'e6-9', source: '6', target: '9' },
    { id: 'e6-10', source: '6', target: '10' },
    { id: 'e7-11', source: '7', target: '11' },
    { id: 'e8-11', source: '8', target: '11' },
    { id: 'e9-11', source: '9', target: '11' },
    { id: 'e10-11', source: '10', target: '11' },
  ];

  const paths: LearningPath[] = [
    {
      id: 'path1',
      name: 'Complete Beginner',
      description: 'Learn everything from scratch',
      duration: '2 months',
      difficulty: 'beginner',
      nodeIds: ['1', '2', '3', '4', '5', '7', '11'],
    },
    {
      id: 'path2',
      name: 'Modern React',
      description: 'Focus on hooks and functional components',
      duration: '6 weeks',
      difficulty: 'intermediate',
      nodeIds: ['1', '4', '6', '8', '9', '11'],
    },
    {
      id: 'path3',
      name: 'Full Stack Focus',
      description: 'Emphasize routing and state management',
      duration: '8 weeks',
      difficulty: 'intermediate',
      nodeIds: ['1', '2', '4', '6', '9', '10', '11'],
    },
  ];

  return { nodes, edges, paths };
}

function generateDataStructuresGraph(): SkillGraphData {
  const nodes: Node[] = [
    {
      id: '1',
      type: 'input',
      data: { label: 'Data Structures' },
      position: { x: 400, y: 0 },
    },
    {
      id: '2',
      data: { label: 'Arrays & Strings' },
      position: { x: 200, y: 100 },
    },
    {
      id: '3',
      data: { label: 'Big O Notation' },
      position: { x: 600, y: 100 },
    },
    {
      id: '4',
      data: { label: 'Linked Lists' },
      position: { x: 100, y: 220 },
    },
    {
      id: '5',
      data: { label: 'Stacks & Queues' },
      position: { x: 350, y: 220 },
    },
    {
      id: '6',
      data: { label: 'Hash Tables' },
      position: { x: 600, y: 220 },
    },
    {
      id: '7',
      data: { label: 'Trees' },
      position: { x: 250, y: 340 },
    },
    {
      id: '8',
      data: { label: 'Graphs' },
      position: { x: 550, y: 340 },
    },
    {
      id: '9',
      data: { label: 'Heaps' },
      position: { x: 100, y: 460 },
    },
    {
      id: '10',
      type: 'output',
      data: { label: 'Advanced DS' },
      position: { x: 400, y: 460 },
    },
  ];

  const edges: Edge[] = [
    { id: 'e1-2', source: '1', target: '2' },
    { id: 'e1-3', source: '1', target: '3' },
    { id: 'e2-4', source: '2', target: '4' },
    { id: 'e2-5', source: '2', target: '5' },
    { id: 'e3-6', source: '3', target: '6' },
    { id: 'e4-7', source: '4', target: '7' },
    { id: 'e5-7', source: '5', target: '7' },
    { id: 'e6-8', source: '6', target: '8' },
    { id: 'e7-9', source: '7', target: '9' },
    { id: 'e7-10', source: '7', target: '10' },
    { id: 'e8-10', source: '8', target: '10' },
    { id: 'e9-10', source: '9', target: '10' },
  ];

  const paths: LearningPath[] = [
    {
      id: 'path1',
      name: 'Fundamentals First',
      description: 'Master the basics before advanced structures',
      duration: '6 weeks',
      difficulty: 'beginner',
      nodeIds: ['1', '2', '3', '4', '5', '7'],
    },
    {
      id: 'path2',
      name: 'Interview Prep',
      description: 'Focus on most commonly asked data structures',
      duration: '4 weeks',
      difficulty: 'intermediate',
      nodeIds: ['1', '2', '4', '6', '7', '8', '10'],
    },
  ];

  return { nodes, edges, paths };
}

function generateWebDevGraph(): SkillGraphData {
  const nodes: Node[] = [
    {
      id: '1',
      type: 'input',
      data: { label: 'Web Development' },
      position: { x: 400, y: 0 },
    },
    {
      id: '2',
      data: { label: 'HTML Basics' },
      position: { x: 150, y: 100 },
    },
    {
      id: '3',
      data: { label: 'CSS Fundamentals' },
      position: { x: 400, y: 100 },
    },
    {
      id: '4',
      data: { label: 'JavaScript Basics' },
      position: { x: 650, y: 100 },
    },
    {
      id: '5',
      data: { label: 'Responsive Design' },
      position: { x: 200, y: 220 },
    },
    {
      id: '6',
      data: { label: 'DOM Manipulation' },
      position: { x: 500, y: 220 },
    },
    {
      id: '7',
      data: { label: 'Frontend Framework' },
      position: { x: 300, y: 340 },
    },
    {
      id: '8',
      data: { label: 'API Integration' },
      position: { x: 600, y: 340 },
    },
    {
      id: '9',
      type: 'output',
      data: { label: 'Full Stack Development' },
      position: { x: 400, y: 460 },
    },
  ];

  const edges: Edge[] = [
    { id: 'e1-2', source: '1', target: '2' },
    { id: 'e1-3', source: '1', target: '3' },
    { id: 'e1-4', source: '1', target: '4' },
    { id: 'e2-5', source: '2', target: '5' },
    { id: 'e3-5', source: '3', target: '5' },
    { id: 'e4-6', source: '4', target: '6' },
    { id: 'e5-7', source: '5', target: '7' },
    { id: 'e6-7', source: '6', target: '7' },
    { id: 'e6-8', source: '6', target: '8' },
    { id: 'e7-9', source: '7', target: '9' },
    { id: 'e8-9', source: '8', target: '9' },
  ];

  const paths: LearningPath[] = [
    {
      id: 'path1',
      name: 'Frontend Focused',
      description: 'Master frontend technologies',
      duration: '3 months',
      difficulty: 'beginner',
      nodeIds: ['1', '2', '3', '5', '4', '6', '7'],
    },
    {
      id: 'path2',
      name: 'Full Stack Journey',
      description: 'Complete web development path',
      duration: '5 months',
      difficulty: 'intermediate',
      nodeIds: ['1', '2', '3', '4', '5', '6', '7', '8', '9'],
    },
  ];

  return { nodes, edges, paths };
}

function generateSpanishGraph(): SkillGraphData {
  const nodes: Node[] = [
    {
      id: '1',
      type: 'input',
      data: { label: 'Spanish Language' },
      position: { x: 400, y: 0 },
    },
    {
      id: '2',
      data: { label: 'Alphabet & Pronunciation' },
      position: { x: 200, y: 100 },
    },
    {
      id: '3',
      data: { label: 'Basic Vocabulary' },
      position: { x: 600, y: 100 },
    },
    {
      id: '4',
      data: { label: 'Present Tense' },
      position: { x: 300, y: 220 },
    },
    {
      id: '5',
      data: { label: 'Common Phrases' },
      position: { x: 600, y: 220 },
    },
    {
      id: '6',
      data: { label: 'Past Tenses' },
      position: { x: 200, y: 340 },
    },
    {
      id: '7',
      data: { label: 'Future Tense' },
      position: { x: 500, y: 340 },
    },
    {
      id: '8',
      type: 'output',
      data: { label: 'Conversational Fluency' },
      position: { x: 400, y: 460 },
    },
  ];

  const edges: Edge[] = [
    { id: 'e1-2', source: '1', target: '2' },
    { id: 'e1-3', source: '1', target: '3' },
    { id: 'e2-4', source: '2', target: '4' },
    { id: 'e3-5', source: '3', target: '5' },
    { id: 'e4-6', source: '4', target: '6' },
    { id: 'e5-6', source: '5', target: '6' },
    { id: 'e4-7', source: '4', target: '7' },
    { id: 'e6-8', source: '6', target: '8' },
    { id: 'e7-8', source: '7', target: '8' },
  ];

  const paths: LearningPath[] = [
    {
      id: 'path1',
      name: 'Tourist Essentials',
      description: 'Learn basics for travel',
      duration: '4 weeks',
      difficulty: 'beginner',
      nodeIds: ['1', '2', '3', '5'],
    },
    {
      id: 'path2',
      name: 'Conversational Spanish',
      description: 'Full grammar and conversation',
      duration: '6 months',
      difficulty: 'intermediate',
      nodeIds: ['1', '2', '3', '4', '5', '6', '7', '8'],
    },
  ];

  return { nodes, edges, paths };
}

function generatePhotographyGraph(): SkillGraphData {
  const nodes: Node[] = [
    {
      id: '1',
      type: 'input',
      data: { label: 'Photography' },
      position: { x: 400, y: 0 },
    },
    {
      id: '2',
      data: { label: 'Camera Basics' },
      position: { x: 200, y: 100 },
    },
    {
      id: '3',
      data: { label: 'Composition Rules' },
      position: { x: 600, y: 100 },
    },
    {
      id: '4',
      data: { label: 'Exposure Triangle' },
      position: { x: 100, y: 220 },
    },
    {
      id: '5',
      data: { label: 'Lighting Techniques' },
      position: { x: 400, y: 220 },
    },
    {
      id: '6',
      data: { label: 'Manual Mode' },
      position: { x: 250, y: 340 },
    },
    {
      id: '7',
      data: { label: 'Post-Processing' },
      position: { x: 550, y: 340 },
    },
    {
      id: '8',
      type: 'output',
      data: { label: 'Advanced Photography' },
      position: { x: 400, y: 460 },
    },
  ];

  const edges: Edge[] = [
    { id: 'e1-2', source: '1', target: '2' },
    { id: 'e1-3', source: '1', target: '3' },
    { id: 'e2-4', source: '2', target: '4' },
    { id: 'e2-5', source: '2', target: '5' },
    { id: 'e4-6', source: '4', target: '6' },
    { id: 'e5-6', source: '5', target: '6' },
    { id: 'e3-7', source: '3', target: '7' },
    { id: 'e6-8', source: '6', target: '8' },
    { id: 'e7-8', source: '7', target: '8' },
  ];

  const paths: LearningPath[] = [
    {
      id: 'path1',
      name: 'Auto to Manual',
      description: 'Graduate from auto mode to manual',
      duration: '6 weeks',
      difficulty: 'beginner',
      nodeIds: ['1', '2', '4', '6'],
    },
    {
      id: 'path2',
      name: 'Complete Photographer',
      description: 'Full photography mastery',
      duration: '3 months',
      difficulty: 'intermediate',
      nodeIds: ['1', '2', '3', '4', '5', '6', '7', '8'],
    },
  ];

  return { nodes, edges, paths };
}

function generateGenericGraph(skill: string): SkillGraphData {
  const nodes: Node[] = [
    {
      id: '1',
      type: 'input',
      data: { label: skill },
      position: { x: 400, y: 0 },
    },
    {
      id: '2',
      data: { label: 'Fundamentals' },
      position: { x: 200, y: 120 },
    },
    {
      id: '3',
      data: { label: 'Core Concepts' },
      position: { x: 600, y: 120 },
    },
    {
      id: '4',
      data: { label: 'Basic Techniques' },
      position: { x: 150, y: 240 },
    },
    {
      id: '5',
      data: { label: 'Theory' },
      position: { x: 400, y: 240 },
    },
    {
      id: '6',
      data: { label: 'Practice & Application' },
      position: { x: 650, y: 240 },
    },
    {
      id: '7',
      data: { label: 'Intermediate Skills' },
      position: { x: 250, y: 360 },
    },
    {
      id: '8',
      data: { label: 'Advanced Concepts' },
      position: { x: 550, y: 360 },
    },
    {
      id: '9',
      type: 'output',
      data: { label: 'Mastery' },
      position: { x: 400, y: 480 },
    },
  ];

  const edges: Edge[] = [
    { id: 'e1-2', source: '1', target: '2' },
    { id: 'e1-3', source: '1', target: '3' },
    { id: 'e2-4', source: '2', target: '4' },
    { id: 'e2-5', source: '2', target: '5' },
    { id: 'e3-6', source: '3', target: '6' },
    { id: 'e4-7', source: '4', target: '7' },
    { id: 'e5-7', source: '5', target: '7' },
    { id: 'e6-8', source: '6', target: '8' },
    { id: 'e7-9', source: '7', target: '9' },
    { id: 'e8-9', source: '8', target: '9' },
  ];

  const paths: LearningPath[] = [
    {
      id: 'path1',
      name: 'Beginner Track',
      description: 'Start with fundamentals and build slowly',
      duration: '8 weeks',
      difficulty: 'beginner',
      nodeIds: ['1', '2', '4', '5', '7'],
    },
    {
      id: 'path2',
      name: 'Practical Approach',
      description: 'Focus on hands-on practice',
      duration: '6 weeks',
      difficulty: 'intermediate',
      nodeIds: ['1', '3', '6', '8', '9'],
    },
    {
      id: 'path3',
      name: 'Complete Mastery',
      description: 'Comprehensive learning path',
      duration: '12 weeks',
      difficulty: 'advanced',
      nodeIds: ['1', '2', '3', '5', '6', '7', '8', '9'],
    },
  ];

  return { nodes, edges, paths };
}
