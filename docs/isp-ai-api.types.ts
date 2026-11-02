// API v1.1: the browser calls Cloud Run with a Firebase employee ID token.
export type AdultIspDomain =
  | "身體福祉"
  | "情緒福祉"
  | "物質福祉"
  | "個人發展"
  | "自我決策"
  | "人際關係"
  | "權利"
  | "社會融合";

export interface StudentIdentity {
  studentId: string;
}

export interface AdultIspIdentity extends StudentIdentity {
  ispId: string;
  domain: AdultIspDomain;
  service_type?: "成人";
}

export interface ApiResponse {
  request_id: string;
}

export interface ApiError extends ApiResponse {
  detail: string;
}

export interface ServiceUserItem {
  id: string;
  name: string;
}

export interface ServiceUserListResponse extends ApiResponse {
  service_users: ServiceUserItem[];
}

export interface IspItem {
  id: string;
  name: string;
}

export type AdultIspListRequest = StudentIdentity;

export interface AdultIspListResponse extends ApiResponse {
  basic_info: string;
  isps: IspItem[];
}

export type AdultIspContextRequest = AdultIspIdentity;

export interface AdultIspContextResponse extends ApiResponse {
  basic_info: string;
  summary: string;
  dream: string;
}

export interface LongTermGoalRequest extends AdultIspIdentity {
  custom_important_to?: string[];
  custom_important_for?: string[];
}

export interface LongTermGoalResponse extends ApiResponse {
  important_to: string[];
  important_for: string[];
  mode: string;
}

export interface ReferenceSearchRequest extends AdultIspIdentity {
  selected_long_goal: string;
  top_k?: number;
}

export interface ReferenceCandidate {
  id: string;
  text: string;
  score: number;
  kind: string;
}

export interface ReferenceSearchResponse extends ApiResponse {
  search_direction: string[];
  short_term_references: ReferenceCandidate[];
  strategy_references: ReferenceCandidate[];
  mode: string;
}

export interface FinalGenerationRequest extends AdultIspIdentity {
  selected_long_goal: string;
  selected_short_reference_ids?: string[];
  selected_strategy_reference_ids?: string[];
}

export interface FinalGenerationResponse extends ApiResponse {
  result: string;
  mode: string;
}

export interface RefineGenerationRequest extends AdultIspIdentity {
  original: string;
  refine_prompt: string;
}

export type RefineGenerationResponse = FinalGenerationResponse;
